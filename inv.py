import os


from InvoiceGenerator.api import Invoice, Item, Client, Provider, Creator
from InvoiceGenerator.pdf import SimpleInvoice
from decimal import Decimal
from pdf2image import convert_from_path

# choose english as language
os.environ["INVOICE_LANG"] = "en"

def generate():
    client = Client('Roga&Copyta',
                    address='21 Avery Lane, #28',
                    zip_code='95032',
                    city='Los Gatos, CA',
                    phone='(650)1233478',
                    email='test@intellectsoft.com',
                    country='USA')
    provider = Provider(
        'IntellectSoft',
        address='21 Avery Lane, #28',
        city='Los Gatos, CA',
        zip_code='95032',
        phone='(650)1233478',
        email='test@intellectsoft.com',
        bank_name='BOFA',
        bank_account='2600420569',
        bank_code='2010',
        note='Payments for service',
        logo_filename='./intellectsoft.jpg',
        country='USA'
    )
    #Person Name
    creator = Creator('Alexander Gunin')

    invoice = Invoice(client, provider, creator)
    invoice.number = '232432432'
    invoice.currency_locale = 'en_US.UTF-8'
    invoice.currency='$'
    invoice.add_item(Item(1, 100, description='Install software', unit='', tax=Decimal(15)))
    invoice.add_item(Item(1, 150, description='Support', unit='', tax=Decimal(15)))
    invoice.add_item(Item(1, 1000000, description='OpenText License', unit='', tax=Decimal(15)))
    return invoice

pdf = SimpleInvoice(invoice)
pdf.gen("invoice.pdf", generate_qr_code=False)


pages = convert_from_path('invoice.pdf', 500)
if len(pages)>0:
    for page in pages:
        page.save('invoice.jpg', 'JPEG')


def _parser(example):
    zero = tf.zeros([1], dtype=tf.int64)
    features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape':
            tf.FixedLenFeature(3, tf.int64),
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/label':
            tf.VarLenFeature(tf.int64)
    }
    res = tf.parse_single_example(example, features)
    img = tf.image.decode_jpeg(res['image/encoded'], channels=3)
    original_w = tf.cast(res['image/shape'][1], tf.int32)
    original_h = tf.cast(res['image/shape'][0], tf.int32)
    img = tf.reshape(img, [original_h, original_w, 3])
    ymin = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymin']*original_h), tf.int32)
    xmin = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmin']*original_w), tf.int32)
    xmax = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/xmax']*original_w), tf.int32)
    ymax = tf.cast(tf.sparse_tensor_to_dense(res['image/object/bbox/ymax']*original_h), tf.int32)
    imgs,labels = tf.py_func(
        crop_py,
        [img, ymin, xmin, ymax,xmax,res['image/object/bbox/label']],
        [tf.int32,tf.int32]
    )