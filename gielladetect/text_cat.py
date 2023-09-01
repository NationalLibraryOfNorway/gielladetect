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


import codecs
import glob
import os
import re
import sys

from gielladetect import util

here = os.path.dirname(__file__)


def pretty_tbl(table):
    return ", ".join(f"{k}:{v}" for k, v in table)


def ensure_unicode(text):
    """Make sure text is unicode

    Helper for functions that should be able to operate on either utf-8
    encoded bytes or decoded unicode objects
    """
    if type(text) == bytes:
        return text.decode("utf-8")
    else:
        assert type(text) == str
        return text


class NGramModel:
    SPLITCHARS = re.compile(
        r"[][}{)(>< \n\t:;!.?_,¶§%&£€$¹°½¼¾©←→▪➢√|#–‒…·•@~\\/”“«»\"0-9=*+‑-]"
    )
    NB_NGRAMS = 400
    MISSING_VALUE = 400

    def __init__(self, arg={}, lang="input"):
        self.lang = lang  # for debugging
        self.unicode_warned = 0

    def of_text(self, text):
        self.finish(self.freq_of_text(ensure_unicode(text), {}))
        return self

    def of_freq(self, freq):
        self.finish(freq)
        return self

    def of_text_file(self, fil):
        self.finish(self.freq_of_text_file(fil))
        return self

    def of_model_file(self, fil, fname):
        raise NotImplementedError("You have to subclass and override of_model_file")

    def freq_of_model_file(self, fil, fname, gram_column, freq_column):
        freq = {}
        for nl, strline in enumerate(fil.readlines()):
            line = strline.strip()
            if line == "":
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    "%s:%d invalid line, was split to %s" % (fname, nl + 1, parts)
                )
            try:
                g = parts[gram_column]
                f = int(parts[freq_column])
                freq[g] = f
            except ValueError as e:
                raise ValueError("%s: %d %s" % (fname, nl + 1, e))
        return freq

    def tokenise(self, text):
        """Tokenise the text

        Since we use split() when loading the model file, we also use split()
        on the input text; this includes whitespace (like byte order
        marks) that might not all be in SPLITCHARS
        """
        tokens = (re.split(self.SPLITCHARS, t) for t in text.split())
        return sum(tokens, [])  # flatten

    def freq_of_text(self, text, freq):
        """This should update freq and return it."""
        raise NotImplementedError("You have to subclass and override freq_of_text")

    def to_model_file(self, fil, fname):
        raise NotImplementedError("You have to subclass and override to_model_file")

    def freq_of_text_file(self, fil):
        freq = {}
        for nl, strline in enumerate(fil.readlines()):
            try:
                line = strline
            except UnicodeDecodeError as e:
                if self.unicode_warned == 0:
                    util.note(
                        "WARNING: Line {} gave {}, skipping ... "
                        "(not warning again)".format(nl, e)
                    )
                self.unicode_warned += 1
                continue
            freq = self.freq_of_text(line, freq)
        if self.unicode_warned != 0:
            util.note(f"Saw {self.unicode_warned} UnicodeDecodeErrors")
        return freq

    def finish(self, freq):
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

    def compare(self, unknown):
        missing_count = len(unknown.ngramskeyset - self.ngramskeyset)
        d_missing = self.MISSING_VALUE * missing_count
        d_found = sum(
            abs(rank - self.ngrams[gram])
            for gram, rank in unknown.ngrams.items()
            if gram in self.ngrams
        )

        return d_missing + d_found


class CharModel(NGramModel):
    def of_model_file(self, fil, fname):
        self.finish(self.freq_of_model_file(fil, fname, gram_column=0, freq_column=1))
        return self

    def to_model_file(self, fil):
        lines = "".join(
            [
                "%s\t%d\n" % (g, f)
                for g, f in util.sort_by_value(self.freq, reverse=True)
                if g != ""
            ]
        )
        fil.write(lines)

    def freq_of_text(self, text, freq):
        words = self.tokenise(text)
        for word in words:
            _word_ = "_" + word + "_"
            size = len(_word_)
            for i in range(size):
                for s in (1, 2, 3, 4):
                    sub = _word_[i : i + s]
                    freq[sub] = freq.get(sub, 0) + 1
                    if i + s >= size:
                        break
        return freq


class WordModel(NGramModel):
    NB_NGRAMS = 30000

    def of_model_file(self, fil, fname):
        self.finish(self.freq_of_model_file(fil, fname, gram_column=1, freq_column=0))
        return self

    def to_model_file(self, fil):
        lines = "".join(
            [
                "%d\t%s\n" % (f, g)
                for g, f in util.sort_by_value(self.freq, reverse=True)
                if g != ""
            ]
        )
        fil.write(lines)

    def freq_of_text(self, text, freq):
        words = self.tokenise(text)
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        return freq

    def finish(self, freq):
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

    def compare_tc(self, unknown_text, normaliser):
        """Implements line 442 of text_cat.pl

        `normaliser` is results[language] from CharModel
        """
        if normaliser <= 0:
            return normaliser
        else:
            unknown_freq = self.freq_of_text(unknown_text, {})
            return sum(
                self.invrank[word] ** 2 * unknown_freq[word] * 100 / normaliser
                for word in unknown_freq.keys()
                if word in self.ngrams
            )


class Classifier:
    """Guess which language a text is written in."""

    DROP_RATIO = 1.10

    def __init__(self, folder=None, langs=[], verbose=False):
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
        else:
            self.langs = set(self.cmodels.keys())
            self.langs_warned = set()

    def get_langs(self, langs=None):
        """Get the set of wanted languages.

        Args:
            langs (None|list[str]): list of probable languages

        Returns:
            (set[str]): The set of languages that should be considered
        """
        if not langs:
            return self.langs
        else:
            langs = set(langs)
            active_langs = self.langs & langs
            if len(langs) != len(active_langs):
                missing = langs - active_langs - self.langs_warned
                if missing:
                    # only warn once per lang
                    self.langs_warned.update(missing)
                    util.note(
                        "WARNING: No language model for {}".format("/".join(missing))
                    )
            return active_langs

    def classify_full(self, intext, langs=[], verbose=False):
        active_langs = self.get_langs(langs)

        text = ensure_unicode(intext)
        ingram = CharModel().of_text(text)

        cscored = {
            l: model.compare(ingram)
            for l, model in self.cmodels.items()
            if l in active_langs
        }
        cranked = util.sort_by_value(cscored)
        cbest = cranked[0]
        cfiltered = {l: d for l, d in cranked if d <= cbest[1] * self.DROP_RATIO}

        if len(cfiltered) <= 1:
            if verbose:
                util.note(f"lm gave: {cfiltered} as only result for input: {text}")
            return list(cfiltered.items())
        else:
            # Along with compare_tc, implements text_cat.pl line
            # 442 and on:
            wscored = {
                l: model.compare_tc(text, cscored[l])
                for l, model in self.wmodels.items()
                if l in cfiltered
            }
            cwcombined = {l: (cscored[l] - wscore) for l, wscore in wscored.items()}
            cwranked = util.sort_by_value(cwcombined)
            if verbose:
                if cranked[: len(cwranked)] == cwranked:
                    util.note(
                        "lm gave: {}\t\twm gave no change\t\tfor"
                        "input: {}".format(pretty_tbl(cranked), text)
                    )
                else:
                    util.note(
                        "lm gave: {}\t\twm-weighted to: "
                        "{}\t\tfor input: {}".format(
                            pretty_tbl(cranked), pretty_tbl(cwranked), text
                        )
                    )
            return cwranked

    def classify(self, text, langs=[], verbose=False):
        return self.classify_full(text, langs, verbose)[0][0]


_classifier = Classifier()


def detect(text, langs=[]):
    return _classifier.classify(text, langs=langs)
