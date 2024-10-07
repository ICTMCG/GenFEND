import json

data_path = './Weibo21/' # './role_virtual_comments/'
file_list = ['train.json', 'test.json', 'val.json']
if __name__ == '__main__':
    new_items = []
    for file_name in file_list:
        with open(data_path + file_name, 'r', encoding = 'UTF-8') as f:
            items = json.load(f)
            comment_num = 0
            file_comment_num = 0
            for index, item in enumerate(items):
                if index < (len(items) / 100) * 100:
                    file_index = int(index / 100) + 1
                    file_index = int(file_index * 100)
                else:
                    file_index = len(items)
                
                from_index = comment_num - file_comment_num
                if "all_comments" in item.keys():
                    to_index = from_index + len(item['all_comments'])
                    comment_num += len(item['all_comments'])
                if "comments" in item.keys():
                    to_index = from_index + len(item['comments'])
                    comment_num += len(item['comments'])
                if (index + 1) % 100 == 0:
                    file_comment_num = comment_num
                item['file_index'] = file_index
                item['from_index'] = from_index
                item['to_index'] = to_index
                new_items.append(item)
        f.close()
        if 'train' in file_name:
            with open(data_path + 'train_index.json', 'w', encoding = 'UTF-8') as f:
                json.dump(new_items, f, ensure_ascii = False, indent = 4)
            f.close() 
        elif 'val' in file_name:
            with open(data_path + 'val_index.json', 'w', encoding = 'UTF-8') as f:
                json.dump(new_items, f, ensure_ascii = False, indent = 4)
            f.close()
        elif 'test' in file_name:
            with open(data_path + 'test_index.json', 'w', encoding = 'UTF-8') as f: 
                json.dump(new_items, f, ensure_ascii = False, indent = 4)
            f.close()   