#!/usr/bin/env python

#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this file. If not, see <http://www.gnu.org/licenses/>.
#
#   Copyright © 2014-2023 The University of Tromsø & the Norwegian Sámi Parliament
#   http://giellatekno.uit.no & http://divvun.no
#
#   ADDENDUM: Pierre Beauguitte, National Library of Norway, 2023
#   Change `langs is None` with `not langs` in Classifier.get_langs
#   Keep only methods used for classification
#   ADDENDUM: Magnus Breder Birkenes, National Library of Norway, 2023
#   Updated tokenise() with a new flatten function, using itertools chain - much faster
#
"""An implementation of the ``N-Gram-Based Text Categorization'' algorithm.

Original article:

Cavnar, W. B. and J. M. Trenkle, ``N-Gram-Based Text
Categorization'' In Proceedings of Third Annual Symposium on
Document Analysis and Information Retrieval, Las Vegas, NV, UNLV
Publications/Reprographics, pp. 161-175, 11-13 April 1994.

Original Perl implementation and article available from
http://odur.let.rug.nl/~vannoord/TextCat/
"""


from typing import Optional, Self, TextIO, Tuple
import codecs
import glob
import os
import re
import itertools

from gielladetect import util

here = os.path.dirname(__file__)


def pretty_tbl(table: list[Tuple[str, float]]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in table)


def ensure_unicode(text: str) -> str:
    """Make sure text is unicode

    Helper for functions that should be able to operate on either utf-8
    encoded bytes or decoded unicode objects
    """
    if isinstance(text, bytes):
        return text.decode("utf-8")
    assert isinstance(text, str)
    return text


class NGramModel:
    SPLITCHARS = re.compile(
        r"[][}{)(>< \n\t:;!.?_,¶§%&£€$¹°½¼¾©←→▪➢√|#–‒…·•@~\\/”“«»\"0-9=*+‑-]"
    )
    NB_NGRAMS = 400
    MISSING_VALUE = 400

    def __init__(self, lang: str = "input") -> None:
        self.lang = lang  # for debugging
        self.unicode_warned = 0

    def of_text(self, text: str) -> Self:
        self.finish(self.freq_of_text(ensure_unicode(text), {}))
        return self

    def of_freq(self, freq: dict[str, float]) -> Self:
        self.finish(freq)
        return self

    def of_text_file(self, fil: TextIO) -> Self:
        self.finish(self.freq_of_text_file(fil))
        return self

    def of_model_file(self, fil: TextIO, fname: str) -> Self:
        raise NotImplementedError("You have to subclass and override of_model_file")

    def freq_of_model_file(self, fil: TextIO, fname: str,
                           gram_column: int, freq_column: int) -> dict[str, float]:
        freq: dict[str, float] = {}
        for nl, strline in enumerate(fil.readlines()):
            line = strline.strip()
            if line == "":
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{fname}:{nl+1} invalid line, was split to {parts}"
                )
            try:
                g = parts[gram_column]
                f = int(parts[freq_column])
                freq[g] = f
            except ValueError as e:
                raise ValueError(f"{fname}: {nl+1} {e}") from e
        return freq

    def tokenise(self, text: str) -> list[str]:
        """Tokenise the text

        Since we use split() when loading the model file, we also use split()
        on the input text; this includes whitespace (like byte order
        marks) that might not all be in SPLITCHARS
        """
        tokens = (re.split(self.SPLITCHARS, t) for t in text.split())
        return list(itertools.chain.from_iterable(tokens))

    def freq_of_text(self, text: str, freq: dict[str, float]) -> dict[str, float]:
        """This should update freq and return it."""
        raise NotImplementedError("You have to subclass and override freq_of_text")

    def freq_of_text_file(self, fil: TextIO) -> dict[str, float]:
        freq: dict[str, float] = {}
        for nl, strline in enumerate(fil.readlines()):
            try:
                line = strline
            except UnicodeDecodeError as e:
                if self.unicode_warned == 0:
                    util.note(
                        f"WARNING: Line {nl} gave {e}, skipping ... "
                        "(not warning again)"
                    )
                self.unicode_warned += 1
                continue
            freq = self.freq_of_text(line, freq)
        if self.unicode_warned != 0:
            util.note(f"Saw {self.unicode_warned} UnicodeDecodeErrors")
        return freq

    def finish(self, freq: dict[str, float]) -> None:
        self.ngrams = {
            gram: rank
            for rank, (gram, freq) in enumerate(
                util.sort_by_value(freq, reverse=True)[: self.NB_NGRAMS]
            )
            if gram != ""
        }
        # Only store the top NB_NGRAMS with frequency:
        self.freq = {gram: freq[gram] for gram in self.ngrams}
        self.ngramskeyset = set(self.ngrams.keys())

    def compare(self, unknown: Self) -> float:
        missing_count = len(unknown.ngramskeyset - self.ngramskeyset)
        d_missing = self.MISSING_VALUE * missing_count
        d_found = sum(
            abs(rank - self.ngrams[gram])
            for gram, rank in unknown.ngrams.items()
            if gram in self.ngrams
        )

        return d_missing + d_found


class CharModel(NGramModel):
    def of_model_file(self, fil: TextIO, fname: str) -> Self:
        self.finish(self.freq_of_model_file(fil, fname, gram_column=0, freq_column=1))
        return self

    def freq_of_text(self, text: str, freq: dict[str, float]) -> dict[str, float]:
        words = self.tokenise(text)
        for word in words:
            _word_ = "_" + word + "_"
            size = len(_word_)
            for i in range(size):
                for s in (1, 2, 3, 4):
                    sub = _word_[i: i + s]
                    freq[sub] = freq.get(sub, 0) + 1
                    if i + s >= size:
                        break
        return freq


class WordModel(NGramModel):
    NB_NGRAMS = 30000

    def of_model_file(self, fil: TextIO, fname: str) -> Self:
        self.finish(self.freq_of_model_file(fil, fname, gram_column=1, freq_column=0))
        return self

    def freq_of_text(self, text: str, freq: dict[str, float]) -> dict[str, float]:
        words = self.tokenise(text)
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        return freq

    def finish(self, freq: dict[str, float]) -> None:
        super().finish(freq)
        # See text_cat.pl line 642ff; we invert and normalise the
        # ranking to make it possible to use compare_tc where one wm
        # is shorter than the other, e.g. if there is only a small
        # corpus for one language, or if we manually deleted some
        # words:
        n_words = len(self.ngrams)
        normaliser = float(n_words) / float(self.NB_NGRAMS)
        self.invrank = {
            gram: ((n_words - rank) / normaliser) for gram, rank in self.ngrams.items()
        }

    def compare_tc(self, unknown_text: str, normaliser: float) -> float:
        """Implements line 442 of text_cat.pl

        `normaliser` is results[language] from CharModel
        """
        if normaliser <= 0:
            return normaliser
        unknown_freq = self.freq_of_text(unknown_text, {})
        return sum(
            self.invrank[word] ** 2 * freq * 100 / normaliser
            for word, freq in unknown_freq.items()
            if word in self.ngrams
        )


class Classifier:
    """Guess which language a text is written in."""

    DROP_RATIO = 1.10

    def __init__(self, folder: Optional[str] = None, langs: Optional[list[str]] = None,
                 verbose: bool = False) -> None:
        if folder is None:
            folder = os.path.join(here, "lm")
        self.cmodels = {}
        self.wmodels = {}

        ext = ".lm"
        fnames = []

        folder_glob = os.path.join(folder, "*" + ext)
        found_fnames = glob.glob(os.path.normcase(folder_glob))
        if not found_fnames:
            raise ValueError(f"No language files found in {folder}")

        if not langs:
            fnames = found_fnames
        else:
            fnames = [os.path.join(folder, lang + ext) for lang in langs]
            not_found = set(fnames) - set(found_fnames)
            if not_found:
                raise ValueError("Unknown language(s): " + ", ".join(not_found))

        for fname in fnames:
            lang = util.basename_noext(fname, ext)
            with codecs.open(fname, "r", encoding="utf8") as fname_stream:
                self.cmodels[lang] = CharModel(lang).of_model_file(fname_stream, fname)
                if verbose:
                    util.note(f"Loaded {fname}")

            fname_wm = os.path.join(folder, lang + ".wm")
            # fname_wmgz = os.path.join(folder, lang+'.wm.gz')
            if os.path.exists(fname_wm):
                with codecs.open(fname_wm, "r", encoding="utf8") as fname_wm_stream:
                    self.wmodels[lang] = WordModel(lang).of_model_file(
                        fname_wm_stream, fname_wm
                    )
                    if verbose:
                        util.note(f"Loaded {fname_wm}")
            else:
                self.wmodels[lang] = WordModel(lang).of_freq({})

        if not self.cmodels:
            raise ValueError("No character models created!")
        self.langs = set(self.cmodels.keys())
        self.langs_warned: set[str] = set()

    def get_langs(self, langs: Optional[list[str]] = None) -> set[str]:
        """Get the set of wanted languages.

        Args:
            langs (None|list[str]): list of probable languages

        Returns:
            (set[str]): The set of languages that should be considered
        """
        if not langs:
            return self.langs
        langs_set = set(langs)
        active_langs = self.langs & langs_set
        if len(langs_set) != len(active_langs):
            missing = langs_set - active_langs - self.langs_warned
            if missing:
                # only warn once per lang
                self.langs_warned.update(missing)
                util.note(
                    f"WARNING: No language model for {'/'.join(missing)}"
                )
        return active_langs

    def classify_full(self, intext: str, langs: Optional[list[str]] = None,
                      verbose: bool = False) -> list[Tuple[str, float]]:
        active_langs = self.get_langs(langs)

        text = ensure_unicode(intext)
        ingram = CharModel().of_text(text)

        cscored = {
            lang: model.compare(ingram)
            for lang, model in self.cmodels.items()
            if lang in active_langs
        }
        cranked = util.sort_by_value(cscored)
        cbest = cranked[0]
        cfiltered = {lang: d for lang, d in cranked if d <= cbest[1] * self.DROP_RATIO}

        if len(cfiltered) <= 1:
            if verbose:
                util.note(f"lm gave: {cfiltered} as only result for input: {text}")
            return list(cfiltered.items())
        # Along with compare_tc, implements text_cat.pl line
        # 442 and on:
        wscored = {
            lang: model.compare_tc(text, cscored[lang])
            for lang, model in self.wmodels.items()
            if lang in cfiltered
        }
        cwcombined = {lang: (cscored[lang] - wscore) for lang, wscore in wscored.items()}
        cwranked = util.sort_by_value(cwcombined)
        if verbose:
            if cranked[: len(cwranked)] == cwranked:
                util.note(
                    f"lm gave: {pretty_tbl(cranked)}\t\twm gave no change\t\tfor"
                    f"input: {text}"
                )
            else:
                util.note(
                    f"lm gave: {pretty_tbl(cranked)}\t\twm-weighted to: "
                    f"{pretty_tbl(cwranked)}\t\tfor input: {text}"
                )
        return cwranked

    def classify(self, text: str, langs: Optional[list[str]] = None, verbose: bool = False) -> str:
        return self.classify_full(text, langs, verbose)[0][0]


_classifier = Classifier()


def detect(text: str, langs: Optional[list[str]] = None) -> str:
    return _classifier.classify(text, langs=langs)
