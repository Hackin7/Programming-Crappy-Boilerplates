# https://realpython.com/python-import/
import module
import module.submodule as g
import module.submodule.file_code as f

print(module.submodule.file_code.var)
print(module.file_code.var)
print(g.file_code.var)
print(f.var)
