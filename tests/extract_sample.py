import lmdb
import six
from PIL import Image

env = lmdb.open('DocTamperV1-TrainingSet', readonly=True, lock=False, readahead=False, meminit=False)

with env.begin(write=False) as txn:
    imgbuf = txn.get(b'image-000000000')
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf)
    im.save('test_sample.jpg', 'JPEG', quality=90)

print("Saved test_sample.jpg")
