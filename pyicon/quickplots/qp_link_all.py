import glob
import sys
import os
import shutil
from importlib import reload
import datetime
import pyicon as pyic                                                                    
import pyicon.quickplots as pyicqp 
reload(pyic)

path_qps = '../../all_qps/'
flist = glob.glob(path_qps+'*/qp_index.html')
flist.sort()

for fpath in flist:
  name = fpath.split('/')[-2][3:]
  print(fpath, name)

qp = pyicqp.QuickPlotWebsite(
  title='List of all simulations',
  author=os.environ.get('USER'),
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  path_data=path_qps,
  info='ICON ocean simulation',
  fpath_css='./qp_css.css',
  fpath_html=path_qps+'index.html'
  )

shutil.copyfile('./qp_css.css', path_qps+'qp_css.css')

text = ''
# --- add link to pyicon docu
text += '<p><li><a href="pyicon_doc/html/index.html">pyicon documentation</a></>'

# --- add link to experiments
for fpath in flist:
  name = fpath.split('/')[-2][3:]
  fpath = fpath.replace(path_qps,'')
  #print(name)
  text += '<p>'
  text += '<li><a href=\"'+fpath+'\">'+name+'</a>'
  text += '</>\n'

qp.header
qp.toc
qp.main
qp.main = text
qp.footer
qp.write_to_file()
