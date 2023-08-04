from src.utils.preprocessing import Preprocessor

IMG_DIR = "data/images"
GEDI_PATH = "data/gedi/gedi_complete.fth"
LABELS_DIR = "data/labels"
PATCH_INFO_FILE = "data/patch_info.fth"

preprocessor = Preprocessor(
    patch_info_file=PATCH_INFO_FILE,
    img_dir=IMG_DIR,
    labels_dir=LABELS_DIR,
    gedi_file=GEDI_PATH,
)

preprocessor.run()
