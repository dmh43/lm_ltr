import os

from fetchers import get_rows

def main():
  path = './indri/query_params.xml'
  try:
    os.remove(path)
  except OSError:
    pass

  with open(path, 'a+') as fh:
    fh.write('<parameters>\n')
    for i, row in enumerate(get_rows()):
      fh.write('<query>\n')
      fh.write('<number>' + str(i + 1) + '</number>\n')
      fh.write('<text>\n')
      fh.write('#combine( ' + row['query'] + ' )\n')
      fh.write('</text>\n')
      fh.write('</query>\n')
    fh.write('</parameters>\n')


if __name__ == "__main__": main()
