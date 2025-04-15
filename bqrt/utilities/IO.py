"""
IO
--

Input/Output utility tools
"""

import pandas as pd

# 读取filepath单个zip文件，即filepath为zip文件路径，并将zip文件中的全部filetype文件合并
def read_zip(filepath, filetype):

    from zipfile import ZipFile

    zip_file = ZipFile(filepath)
    df = pd.concat([pd.read_csv(zip_file.open(z)) for z in zip_file.infolist() if z.filename.endswith('.'+filetype)])

    return df


# 读取folderpath中的多个个zip文件，并将里面的filetype文件合并
def read_multi_zip(folderpath, filetype):

    import glob
    from zipfile import ZipFile

    df = pd.DataFrame()
    all_zips = glob.glob(folderpath+'/*.zip')

    for zip_file in [ZipFile(i) for i in all_zips]:
        to_add = pd.concat([pd.read_csv(zip_file.open(z)) for z in zip_file.infolist() if z.filename.endswith('.'+filetype)])
    df = pd.concat([df, to_add])

    return df