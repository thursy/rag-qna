# Building Apps with watsonx.ai and Carbon
Repository ini berisikan code dari sebuah webapp yang menggunakan Flask python sebagai backend dan Vanilla Java Script, HTML, CSS sebagai Front-end.
Front-end juga menggukan open source HTML web component yang dibuat oleh IBM, dinamakan [Carbon](https://web-components.carbondesignsystem.com/?path=/docs/introduction-welcome--page).
Aplikasi ini memungkinkan anda melakukan tanya jawab menggunakan dokumen anda sendiri dengan bantuan LLM Model. Metode yang digunakan untuk ini dinamakan [RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG).


Coba secara langsung applikasi yang sudah di deploy [disini](https://rag-qna.1gmk3v763a4v.us-south.codeengine.appdomain.cloud/rag#)

## How this app works? 🚀
Aplikasi ini memungkinkan pengguna melakukan tanya jawab dengan LLM seputar dokumen yang diupload. Berikut adalah kapabilitas dari aplikasi ini: 
1. Pengguna dapat mengupload dokumen ke knowledge repository seperti Milvus dan juga Watson Discovery.
2. Pengguna dapat melakukan tanya jawab dengan memilih LLM model, knowledge repository. Pengguna dapat mengaktifkan fitur streaming untuk mendapatkan jawaban secara streaming. Jika streaming di non-aktifkan informasi tambahan mengenai waktu yang dibutuhkan untuk melakukan retrieval dan generation akan ditampilkan.

### Contoh Pertanyaan
Berikut adalah contoh-contoh pertanyaan apakah aplikasi berjalan dengan semestinya.
- Berapa usia pensiun?
- untuk hari rabu karyawan pakai baju apa?
- Bagaimana aturan THR
Coba tanyakan pertanyaan-pertanyaan lain yang jawabannya mungkin ada atau tidak didokumen.  


### Contoh Dokumen yang bisa diupload
Anda dapat menggunakan dokument anda sendiri untuk mencoba kapabilitas RAG menggunakan LLM model yang berbeda, atau knowledge repository yang berbeda.
Berikut beberapa dokumen yang sudah tersedia di Milvus dan Watson Discovery:
- [Dokumen K3](https://github.com/Client-Engineering-Indonesia/watsonx-incubation-2/blob/main/data/K3.pdf)
- [Dokumen Peraturan Perusahaan](https://github.com/Client-Engineering-Indonesia/watsonx-incubation-2/blob/main/data/Peraturan_Perusahaan.pdf)

Jika anda tidak memiliki dokument, silahkan gunakan beberapa contoh dokument berikut ini.
- [Dokumen Evakuasi bencana alam](https://github.com/Client-Engineering-Indonesia/watsonx-incubation-2/blob/main/data/evakuasi%20bencana%20alam.pdf)
- [Dokumen Perlindungan data](https://github.com/Client-Engineering-Indonesia/watsonx-incubation-2/blob/main/data/Peraturan%20Perlindungan%20Data.pdf)  

Download dokumet tersebut ke local computer anda, dan upload ke webapp.  


## Requirements 🚀
### Teknologi yang digunakan
Untuk membangun webapp ini beberapa SDK digunakan untuk memungkinkan interaksi dengan teknologi atau service yang diperlukan.
1. [watsonx.ai python SDK](https://ibm.github.io/watsonx-ai-python-sdk/)
2. [Milvus python SDK](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
3. [Watson Discovery python SDK](https://github.com/watson-developer-cloud/python-sdk/blob/master/ibm_watson/discovery_v2.py)

Selain menggunakan SDK, kita sebenarnya juga bisa menggunakan API. Berikut adalah link dokumentasi untuk beberapa API yang mungkin berguna untuk anda.
- [watsonx.ai API Documentation](https://cloud.ibm.com/apidocs/watsonx-ai)
- [Watson Discovery API Documentation](https://cloud.ibm.com/apidocs/discovery-data?code=python)
  
### Credentials yang harus di sediakan
Agar web application ini dapat dijalankan dengan baik di local computer ataupun di docker container anda harus menyiapkan credentials dan secret berikut.
- watsonx.ai APIKEY
- watsonx.ai PROJECT_ID
- watsonx.ai URL
- Watson Discovery APIKEY
- Watson Discovery PROJECT_ID
- Watson Discovery URL
- Milvus Host
- Milvus Port
- Milvus Password 
  
## Cara memulai 🚀
Ada beberapa cara untuk mendeploy code yang ada di repository ini untuk menjadi sebuah webapp yang dapat dioperasikan.
Pertama, anda bisa menjalankan code tersebut di local computer anda. Cara kedua, anda bisa mendeploynya dengan Docker Container. 
Penggunakan Docker container menguntungkan karena dapat di-deploy dimana saja dengan mudah.
  
### Running di Local Computer
1. Buka terminal, cd kedalam directory Lab 4.
2. create python virtual environment, berikut adalah contoh jika anda ingin membuat virtual environment bernama 'genai'  
```
python -m venv genai
```  

3. aktifkan python virtual environment yang baru saja dibuat  
```
source genai\bin\activate
```  

4. install dependencies yang ada di requirements.txt  
```
python -m pip install -r requirements.txt
```  

5. export secret dan credentials dengan mengganti semua value yang ada didalam double quote dengan credentials anda. Paste command tersebut di terminal.  
```
export WX_API_KEY="IAM-APIKEY"
export WX_PROJECT_ID="WATSONXAI-PROJECT-ID"
export WX_URL="WATSONXAI-URL"
export WD_API_KEY="WATSON-DISCOVERY-APIKEY"
export WD_PROJECT_ID="WATSON-DISCOVERY-PROJECT-ID"
export WD_URL="WATSON-DISCOVERY-INSTANCE-URL"
export milvus_host="MILVUS-HOST"
export milvus_port="LISTEN-TO-PORT"
export milvus_password="MILVUS-PASSWORD"
```    

6. cd ke folder app dan jalankan aplikasi  
```
cd app
python main.py
```  
7. Buka local `127.0.0.0:8080` atau IP yang muncul di terminal setelah applikasi dijalankan.
  
  
### Running menggunakan Docker Container

1. Buka terminal, cd kedalam directory Lab 4.
2. Build docker image menggunakan docker file  
```
docker build -t qna .
```  
3. Run docker container dengan menjadikan credentials dan secret sebagai environment variable.
```
docker run --env "WX_API_KEY=IAM-APIKEY" \
--env "WX_PROJECT_ID=WATSONXAI-PROJECT-ID" \
--env "WX_URL=WATSONXAI-URL" \
--env "WD_API_KEY=WATSON-DISCOVERY-APIKEY" \
--env "WD_PROJECT_ID=WATSON-DISCOVERY-PROJECT-ID" \
--env "WD_URL=WATSON-DISCOVERY-INSTANCE-URL" \
--env "milvus_host=MILVUS-HOST" \
--env "milvus_port=LISTEN-TO-PORT" \
--env "milvus_password=MILVUS-PASSWORD" \
-p 8080:8080 --name qna-container qna
```  
4. Buka URL deployment.

