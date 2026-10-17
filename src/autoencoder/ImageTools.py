import os
from pathlib import Path

from PIL import Image

from src.autoencoder.Configuration import Configuration
from src.Configuration import Configuration as RWConfiguration

class ImageTools:
    def rescalefolder(self,originfolder,destinationfolder,resolution=(512,512)):
        p = Path(originfolder)

        #[x for x in p.iterdir() if x.is_dir()]
        #[PosixPath('.hg'), PosixPath('docs'), PosixPath
        files = [f for f in p.iterdir() if f.match("*.png") and f.match('*bin*') ] #if f.match("145592*.jpg")
        print(files)
        for file in files:

            origindir, filename = os.path.split(file)
            newname = filename[0:filename.find('r')]
            if os.path.isfile(destinationfolder + newname + "resized.png"):
                continue
            im = Image.open(file)
            im = im.resize(resolution)
            im.save(destinationfolder + newname + "resized.png")


autoencoderconf = Configuration()
rewriterfractalconf = RWConfiguration()
imtools = ImageTools()
imtools.rescalefolder(rewriterfractalconf.filepath,rewriterfractalconf.filepath+"rescaled/")
