AST-PST-Tablerecognizer
----------------------

### Recognize and extract information from the AST-PST tables

### Installation

#### 1. Requirements
- Python >= 3.6
- tesseract
- tesserocr
- opencv
- pandas

#### 2. Copy this repository
```
git clone https://github.com/JKamlah/AST-PST-Tablerecognizer.git
cd AST-PST-Tablerecognizer
```

#### 3. Installation into a Python Virtual Environment

    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

### Process steps

#### Start the process

    $ python3 main.py path/to/image[s]
    
    
#### Examples for template 1 & 2

    $ python3 main.py ./testfiles/

Copyright and License
--------

Copyright (c) 2020 Universit├Ątsbibliothek Mannheim

Project management
 * Moritz Lubcyk
 * Moritz Hennicke

Software development
 * [Jan Kamlah](https://github.com/jkamlah)

**Linefinder** is an OSS. You may use it under the terms of the Apache 2.0 License.
See [LICENSE](./LICENSE) for details.
