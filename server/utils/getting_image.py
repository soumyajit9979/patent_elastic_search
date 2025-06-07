import requests

doc_nums = ['20230091382', '20230158633', '17946528', '18048191']
base_url = 'https://patentimages.storage.googleapis.com/pdfs/US{}A1.pdf'

for doc in doc_nums:
    url = base_url.format(doc)
    response = requests.get(url)
    if response.status_code == 200:
        with open(f'{doc}.pdf', 'wb') as f:
            f.write(response.content)
        print(f'Downloaded: {doc}.pdf')
    else:
        print(f'Failed to download {doc}, status: {response.status_code}')
