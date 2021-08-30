# Bypass Blacklists

def hexify(char):
    return '\\'+f'x{hex(ord(char))[2:]:2}'

def string_hexify(string):
    output = ''
    for i in string:
        output += hexify(i)
    return output

print(string_hexify(input()))
