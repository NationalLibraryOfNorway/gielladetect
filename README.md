This repository simply repackages the [language classification script](https://github.com/giellalt/CorpusTools/tree/main#pytextcat) from the GiellaLT's corpus tools ([GiellaLT's website](https://giellalt.github.io/), [original repo](https://github.com/giellalt/CorpusTools)) as a python module.

The source code as well as the language model files are released under the GPL-3.0 license.

### Installation

```
python3 -m pip install .
```

### Usage

```
import gielladetect

gielladetect.detect("Lurer du på hva som rører seg innenfor veggene til Nasjonalbiblioteket på Solli plass i Oslo?")
# Result: 'nob'

# To restrict detection to a subset of languages:
gielladetect.detect("Lurer du på hva som rører seg innenfor veggene til Nasjonalbiblioteket på Solli plass i Oslo?", ['nob', 'nno', 'eng'])
# Result: 'nob'
```
