import os


def mineru_upload_file_demo():

      import requests
      import os
      token = os.getenv("MINERU_TOKEN")
      url = "https://mineru.net/api/v4/file-urls/batch"
      header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
      }
      data = {
      "files": [
            {"name": "demo.pdf", "data_id": "abcd"}
      ],
      "model_version":"vlm"
      }
      file_path = [r"D:\PycharmProjects\langchain_2603\rag\data\sample.pdf"]
      try:
            response = requests.post(url,headers=header,json=data)
            if response.status_code == 200:
                  result = response.json()
                  print('上传成功:{}'.format(result))
                  batch_id = result['data']['batch_id']
                  if result["code"] == 0:
                        batch_id = result["data"]["batch_id"]
                        urls = result["data"]["file_urls"]
                        print('batch_id:{},urls:{}'.format(batch_id, urls))
                        for i in range(0, len(urls)):
                              with open(file_path[i], 'rb') as f:
                                    res_upload = requests.put(urls[i], data=f)
                                    if res_upload.status_code == 200:
                                          print(f"{urls[i]} 上传成功")
                                    else:
                                          print(f"{urls[i]} 上传失败")
                  else:
                        print('apply upload url failed,reason:{}'.format(result.msg))
                  return batch_id
            else:
                  print(f"请求失败，状态码：{response.status_code}，响应内容：{response.text}")
      except Exception as err:
            print(err)



def mineru_check_result_demo(batch_id):
    import requests
    import time
    token = os.getenv("MINERU_TOKEN")
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    header = {
              "Content-Type": "application/json",
              "Authorization": f"Bearer {token}"
    }

    res = requests.get(url, headers=header)
    while res.json()["data"]['extract_result'][0]['state'] != 'done':
              print('当前状态为running，等待3秒后重试')
              time.sleep(3)
              res = requests.get(url, headers=header)
              print(res.status_code)
              print(res.json()["data"]['extract_result'][0]['state'],end="\n\n=========\n\n")
    print('提取结果为:',res.json()["data"]['extract_result'][0]['full_zip_url'])

if __name__ == '__main__':
      # mineru_upload_file_demo()
      mineru_check_result_demo('ea4465e6-7d46-4f9e-ace8-7a3460ff0a4b')