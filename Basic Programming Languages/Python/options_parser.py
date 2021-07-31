from optparse import OptionParser

parser = OptionParser()
# -h is automatically implemented
parser.add_option("-f", "--file", dest="filename", help="Filename")
options, arguments = parser.parse_args()

print(options, arguments)
print(options.filename)
