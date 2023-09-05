Makes the [language classification script](https://github.com/giellalt/CorpusTools/tree/main#pytextcat) from the GiellaLT's corpus tools available as a python module ([GiellaLT's website](https://giellalt.github.io/), [original repo](https://github.com/giellalt/CorpusTools)).

The source code as well as the language model files are released under the GPL-3.0 license.

### Installation

```
pip install gielladetect
```

### Usage

```
import gielladetect

text = "Lurer du på hva som rører seg innenfor veggene til Nasjonalbiblioteket på Solli plass i Oslo?"

gielladetect.detect(text)
# Result: 'nob'

# To restrict detection to a subset of languages:
gielladetect.detect(text, ['nob', 'nno', 'eng'])
# Result: 'nob'
```

### Supported languages

Using [ISO 639-3](https://iso639-3.sil.org/code_tables/639/data) codes.

| Code | Name                |
|------|---------------------|
| ara  | Arabic              |
| bxr  | Russia Buriat       |
| ckb  | Central Kurdish     |
| dan  | Danish              |
| deu  | German              |
| eng  | English             |
| est  | Estonian            |
| fao  | Faroese             |
| fas  | Persian             |
| fin  | Finnish             |
| fit  | Tornedalen Finnish  |
| fkv  | Kven Finnish        |
| fra  | French              |
| hbs  | Serbo-Croatian      |
| isl  | Icelandic           |
| ita  | Italian             |
| kal  | Kalaallisut         |
| kmr  | Northern Kurdish    |
| koi  | Komi-Permyak        |
| kpv  | Komi-Zyrian         |
| krl  | Karelian            |
| mdf  | Moksha              |
| mhr  | Eastern Mari        |
| mns  | Mansi               |
| mrj  | Western Mari        |
| myv  | Erzya               |
| nno  | Norwegian Nynorsk   |
| nob  | Norwegian Bokmål    |
| olo  | Livvi               |
| pol  | Polish              |
| rmf  | Kalo Finnish Romani |
| rmn  | Balkan Romani       |
| rmu  | Tavringer Romani    |
| rmy  | Vlax Romani         |
| ron  | Romanian            |
| rus  | Russian             |
| sma  | Southern Sami       |
| sme  | Northern Sami       |
| smj  | Lule Sami           |
| smn  | Inari Sami          |
| sms  | Skolt Sami          |
| som  | Somali              |
| spa  | Spanish             |
| swe  | Swedish             |
| tur  | Turkish             |
| udm  | Udmurt              |
| urd  | Urdu                |
| vep  | Veps                |
| vie  | Vietnamese          |
| yid  | Yiddish             |
| yrk  | Nenets              |
