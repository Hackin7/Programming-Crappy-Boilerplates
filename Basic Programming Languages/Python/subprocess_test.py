# Inspired by Learning Python with Raspberry Pi
import subprocess

### Call #######################################################################
subprocess.call("ls")
subprocess.call(["cat", "/proc/cpuinfo"])

### Popen ######################################################################
print("#"*80) ##################################################################
p = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
text = p.stdout.read().decode()
error = p.stderr.read().decode()
print(text.split("\n")[0])
print("Error:",error)

### Run ########################################################################
print("#"*80) ##################################################################
result = subprocess.run(
    ["python", "-c", "import sys; print(sys.stdin.read())"], input=b"underwater"
)
print(result)

print("#"*80) ##################################################################
result = subprocess.run(
    ["python", "-c", "import sys; print(sys.stdin.read()); raise Exception('Potential Error')"], input=b"underwater lol",
     stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
text = result.stdout
error = result.stderr
print(text)
print("Error:",error)

print("#"*80) ##################################################################
# Interactive Processes just stop
result = subprocess.run(
    ["cat"], input=b"underwater lol\n 123",
     stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
text = result.stdout
error = result.stderr
print(text)
print("Error:",error)
