from tool import dataset_tool

if __name__ == '__main__':
    # file_name = '/root/olmo_datasets_tool/data/RedPajamaWikipedia/chunk_98.jsonl.zst'
    file_name = '/root/olmo_datasets_tool/data/test/c-sharp_train-00000-of-00045.zst'

    gen = dataset_tool.DataTool.extract_content_from_zstd(file_name, 'text')
    for value in gen:
        print(value)
