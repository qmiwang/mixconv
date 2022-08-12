import nrrd
import SimpleITK as sitk


def read_label(anno_path):
    nrrd_data, nrrd_options = nrrd.read(anno_path)
    return nrrd_data


def read_cta(img_path, resize_scale=1, rtn_spacing=False):  # 读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
    reader = sitk.ImageSeriesReader()
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(img_path)  # 根据文件夹获取序列ID,一个文件夹里面通常是一个病人的所有切片，会分为好几个序列
    dicom_names = reader.GetGDCMSeriesFileNames(img_path, series_IDs[0])  # 选取其中一个序列ID,获得该序列的若干文件名
    reader.SetFileNames(dicom_names)  # 设置文件名

    # reader.ReadImageInformation()

    # print(reader.GetMetaData('0028|1050'))

    image3D = reader.Execute()  # 读取dicom序列
    spacing = image3D.GetSpacing()
    direction = image3D.GetDirection()
    origin = image3D.GetOrigin()
    image3D = sitk.GetArrayFromImage(image3D)  # z, y, x
    if not rtn_spacing:
        return image3D
    else:
        return image3D, spacing, direction, origin