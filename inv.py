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