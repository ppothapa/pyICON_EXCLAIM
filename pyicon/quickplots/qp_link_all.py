import glob
import sys
import os
import shutil
from importlib import reload
import datetime
import pyicon as pyic
import pyicon.quickplots as pyicqp 
reload(pyic)

# --- search path
path_qps = '../../all_qps/'

if len(sys.argv)==2:
  path_qps  = sys.argv[1]
  if not path_qps.endswith('/'):
    path_qps += '/'
  run = path_qps.split('/')[-1]
  title = f'List of timeaverages for {run}'
  fname_html = 'qp_index.html'
  top_level = False
else:
  top_level = True
  title='List of all simulations'
  fname_html = 'index.html'

# --- find all pags that should be linked
flist = glob.glob(path_qps+'*/qp_index.html')
flist.sort()

print('qp_link_all: ',flist)

# --- make header
qp = pyicqp.QuickPlotWebsite(
  title=title,
  author=os.environ.get('USER'),
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  path_data=path_qps,
  info='ICON data plotted by pyicon.quickplots',
  fpath_css='./qp_css.css',
  fpath_html=path_qps+fname_html
  )

# --- copy css file
shutil.copyfile('./qp_css.css', path_qps+'qp_css.css')

# --- start with content
text = ''
# --- add link to pyicon docu
if top_level:
  text += '<p><li><a href="pyicon_doc/html/index.html">pyicon documentation</a></>\n'
# --- add link to experiments / timeaverages
for fpath in flist:
  name = fpath.split('/')[-2]#[3:]
  name = name.replace('qp-','')
  fpath = fpath.replace(path_qps,'')
  print(fpath, name)
  text += '<p>'
  text += '<li><a href=\"'+fpath+'\">'+name+'</a>'
  text += '</>\n'
qp.main = text

# --- finally put everything together
qp.write_to_file()

# --- for diagnostics
if False:
  print(qp.header)
  print(qp.toc)
  print(qp.main)
  print(qp.footer)
