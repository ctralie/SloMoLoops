import glob
import subprocess

if __name__ == '__main__':
    for f in glob.glob("*.svg"):
        fnew = f[0:-4] + ".pdf"
        subprocess.call(["inkscape", "-D", "-z", "--file=%s"%f, "--export-pdf=%s"%fnew])
