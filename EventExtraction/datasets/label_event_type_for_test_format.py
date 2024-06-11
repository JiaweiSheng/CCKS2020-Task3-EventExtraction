import json


# TYPES = ['起诉', '投资', '减持', '股份股权转让', '质押', '收购', '判决']

def main():
    with open('tagger_share/args_test.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_types = []
    data_texts = []
    for line in lines:
        # print(line)
        data_id, content = line.strip().split('\t')
        s1, s2 = content.split('_#_#_')
        data_type = ''.join(s1[4:].split('_'))
        data_text = ''.join(s2.split('_'))
        # print(data_type, data_text)
        if data_type == '股份股权转让':
            data_type = '转让'
        data_ids.append(data_id)
        data_types.append(data_type)
        data_texts.append(data_text)

    with open('testing_data/test_format.json', 'w', encoding='utf-8') as f:
        for i in range(len(data_ids)):
            data = {'id': data_ids[i],
                    'type': data_types[i],
                    'content': data_texts[i]}
            data_json = json.dumps(data, ensure_ascii=False)
            f.write(data_json + '\n')


if __name__ == '__main__':
    main()
