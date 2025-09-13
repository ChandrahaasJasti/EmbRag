import faiss
import os
from pathlib import Path
import json
from trafilatura import fetch_url, extract
import pymupdf4llm
import numpy as np 
import requests

""" things remaining : chunker,faiss """
class EmbRag:
    # cache=""
    # urls=[]
    # files=[]
    def __init__(self,docs_path,faiss_path):
        """faiss index creation"""
        
        index_path =Path(faiss_path+"/index.bin")
        if(index_path.exists()):
            index=faiss.read_index(str(index_path))
        else:
            index=faiss.IndexFlatL2(768)
        self.docs=docs_path
        self.faiss_path=faiss_path
        flag=True
        self.pth=os.path.join(faiss_path,"cache.json")
        self.pth2=os.path.join(faiss_path,"meta_data.json")
        self.pth_checker1(self.pth2)
        self.pth_checker(self.pth)
        with open(self.pth,'r') as f:
            self.cache=json.load(f)
        self.files=os.listdir(docs_path)
        with open(self.pth2,'r') as f:
            self.chunks=f.read()
        l=eval(self.chunks)
        for i in self.files:
            if i not in self.cache:
                if((i.endswith('.txt') or i.endswith('.md')) and not i.startswith('url')):
                    with open(os.path.join(self.docs,i),'r') as f:
                        text=f.read()
                    chunks=self.chunk_text(text)
                    embeds=[]
                    for k in range(len(chunks)):
                        dic={}
                        dic['doc']=i
                        dic['id']=k
                        dic['content']=chunks[k]
                        l.append(dic)
                        embeds.append(self.get_embedding(chunks[k]))
                    ans=np.stack(embeds)
                    index.add(ans)
                    #l.append(chk ) append all the chunks to this list

                elif(i.endswith('.pdf')):
                    md_text = pymupdf4llm.to_markdown(os.path.join(self.docs,i))
                    chunks=self.chunk_text(md_text)
                    embeds=[]
                    for k in range(len(chunks)):
                        dic={}
                        dic['doc']=i
                        dic['id']=k
                        dic['content']=chunks[k]
                        l.append(dic)
                        embeds.append(self.get_embedding(chunks[k]))
                    ans=np.stack(embeds)
                    index.add(ans)
                elif(i.endswith('.txt') and i.startswith('url')):
                    with open(os.path.join(self.docs,i),'r') as f:
                        links=f.read()
                        self.urls=links.split(',')
                    chunks=[]
                    for j in self.urls:
                        downloaded = fetch_url(j)
                        result = extract(downloaded)
                        if result is None:
                            print(f"could not access the url {j} because of authentication ")
                        else:
                            chunks.append(result)
                            l.append({'doc': f"url{j}", "content": result})
                    embeds = []
                    for k in range(len(chunks)):
                        dic = {}
                        dic['doc'] = i
                        dic['id'] = k
                        dic['content'] = chunks[k]
                        l.append(dic)
                        embeds.append(self.get_embedding(chunks[k]))
                    ans=np.stack(embeds)
                    index.add(ans)

                else:
                    flag=False
                    print(f"{i} is not a part of [pdf,txt,website] markitdown feature coming soon")
                self.cache[i]="True"
        with open(self.pth,'w') as f:
            json.dump(self.cache,f,indent=4)
        with open(self.pth2,'w') as f:
            json.dump(l,f,indent=4)
        faiss.write_index(index, str(index_path))
    
    def pth_checker(self,arge):
        file_path=Path(arge)
        if(not file_path.exists()):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path,'w') as f:
                json.dump({},f)
    
    def pth_checker1(self,arge):
        file_path=Path(arge)
        if(not file_path.exists()):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path,'w') as f:
                json.dump([],f)

    def chunk_text(self,text):
        WORD_COUNT = 512
        OVERLAP = 50
        
        # Split text into words
        words = text.split()
        chunks = []
        
        # Calculate the step size (chunk size - overlap)
        step = WORD_COUNT - OVERLAP
        
        # Create chunks with overlap
        for i in range(0, len(words), step):
            # Get the chunk of words
            chunk = words[i:i + WORD_COUNT]
            
            # Only add chunk if it's not empty
            if chunk:
                # Join words back into text
                chunk_text = ' '.join(chunk)
                chunks.append(chunk_text)
                
                # If we've reached the end of the text, break
                if i + WORD_COUNT >= len(words):
                    break
        
        return chunks
    
    def get_embedding(self,text):
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)
    
    def queryDB(self,q):
        vec=self.get_embedding(q).reshape(1,-1)
        index_path =Path(self.faiss_path+"/index.bin")
        if(index_path.exists()):
            index=faiss.read_index(str(index_path))
            D,I=index.search(vec,k=3)
            with open(self.pth2,'r') as f:
                lst=f.read()
            lst=eval(lst)
            indices=I[0]
            ans=[]
            for i in range(len(indices)):
                if indices[i]!=-1:
                    ans.append(lst[i])
            return ans
        else:
            print("no faiss index found")
            ans=[-1]
            return ans