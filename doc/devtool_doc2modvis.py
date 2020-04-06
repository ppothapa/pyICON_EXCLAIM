#!/bin/bash

make html

#rsync -rhv --delete --update doc/_build/html modvis.dkrz.de:/var/www/projects/mh0033/m300602/pyicon_doc
rsync -rhv --delete --update ./_build/html modvis.dkrz.de:/var/www/projects/mh0033/m300602/pyicon_doc
rsync -rhv --delete --update ./_build/html ../all_qps/pyicon_doc
