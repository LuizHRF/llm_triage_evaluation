from time import sleep
from pdf2image import convert_from_bytes
import os
import base64
import shutil
from io import BytesIO
import pdfplumber
from google import genai
from google.genai import types
import sys
import re

def save_images(pdf_file, file_name):

    os.makedirs("images",   exist_ok=True)
    os.makedirs(f"images/{file_name}",  exist_ok=True)
    for entry in os.listdir(f"images/{file_name}"):
        path = os.path.join(f"images/{file_name}", entry)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception:
            pass

    images = convert_from_bytes(pdf_file.read())
    
    for i, image in enumerate(images):

        image.save(f'images/{file_name}/output_page_{i+1}.jpeg', 'JPEG') 

def encode_images(path):

    encoded_images = []

    all_files = os.listdir(path)
    for filename in all_files:
    # Create the full file path.
        image_path = os.path.join(path, filename)

        if os.path.isfile(image_path) and (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            
            with open(image_path, "rb") as image_file:

                encoded_string = image_file.read()

                encoded_images.append(types.Part.from_bytes(
                    data=encoded_string,
                    mime_type='image/jpeg'
                ))

    return encoded_images

def extract_pdf_pages(pdf_file, filename):
    try:
        pdf_file.seek(0)
    except Exception:
        pass
    try:
        data = pdf_file.read()
        if isinstance(data, str):
            data = data.encode()
    except Exception:
        raise ValueError("Não foi possível ler o arquivo PDF.")

    pdf_text = ""
    os.makedirs("texts", exist_ok=True)
    os.makedirs(f"texts/{filename}", exist_ok=True)
    pages = []
    if data:
        try:
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        idx = len(pages)
                        txt_path = os.path.join(f"texts/{filename}", f"page_{idx}.txt")
                        try:
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(t)
                        except Exception:
                            pass
        except Exception:

            raise ValueError("Não foi possível processar o arquivo PDF.")

def create_content_images_with_pdf(encoded_images, filename):

    try:
        pages = []
        texts_dir = os.path.join("texts", filename)

        if os.path.isdir(texts_dir):
            txt_files = [f for f in os.listdir(texts_dir) if f.lower().endswith(".txt")]
            def _sort_key(name):
                m = re.search(r'(\d+)', name)
                return int(m.group(1)) if m else name
            
            txt_files.sort(key=_sort_key)
            for fname in txt_files:
                file_path = os.path.join(texts_dir, fname)
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        pages.append(fh.read())
                except Exception:
                    pages.append("")
        else:
            pages = []
    except Exception:
        pages = []

    content = []
    for i, ei in enumerate(encoded_images):
        if i < 3: continue
        #if i != 10: continue

        #print(pages[i])

        content.append(ei)
        content.append(pages[i])

    return content

client = genai.Client()

content = []
content.append("A seguir vocêm tem uma série de imagens e textos. As imagens são de um protocolo de triagem de pacientes e, após cada imagem, está o texto correspondente extraído do PDF. Extraia o texto das imagens, utilizando as informações do texto do PDF para melhorar a extração. Relacione ao texto as cores que aparecem nas imagens quando for pertinente. Não utilize markdwon, responda em texto comum (títulos, subtítulos e etc.). Responda somente com o texto resultante. Identifique a troca de páginas com um traço entre duas linhas novas.")

if len(sys.argv) < 2:
    raise ValueError("Usage: python parse_protocol.py <pdf_path>")

pdf_path = sys.argv[1]

with open(pdf_path, "rb") as f:
    pdf_file = BytesIO(f.read())

#save_images(pdf_file, "protocolo")
#extract_pdf_pages(pdf_file, "protocolo")
encoded_images = encode_images("images/protocolo")
image_pdf_content = create_content_images_with_pdf(encoded_images, "protocolo")

content.extend(image_pdf_content)

while(True):
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content,
        )
    except Exception as e:
        print(f"Erro ao gerar conteúdo: {e}. Tentando novamente em 1 minuto.")
        sleep(60)
        continue
    break

try:
    text = response.text
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved output to output.txt")
except AttributeError:
    text = response.output[0].content[0].text
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved output to output.txt")