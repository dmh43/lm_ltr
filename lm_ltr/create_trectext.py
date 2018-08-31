from fetchers import get_rows

def main():
  path = './indri/in'
  try:
    os.remove(path)
  except OSError:
    pass
  with open(path, 'a+') as fh:
    for i, row in enumerate(get_rows()):
      fh.write('<DOC>\n')
      fh.write('<DOCNO>' + str(i + 1) + '</DOCNO>\n')
      fh.write('<TEXT>\n')
      fh.write(row['document'] + '\n')
      fh.write('</TEXT>\n')
      fh.write('</DOC>\n')


if __name__ == "__main__": main()
