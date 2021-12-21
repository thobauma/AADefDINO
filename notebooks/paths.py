from pathlib import Path
import getpass

username = getpass.getuser()

DATA_PATH = Path('cluster', 'scratch', username, 'dl_data')
DN_PATH = Path(DATA_PATH, 'DAmageNet')
DND_PATH = Path(DN_PATH, 'DAmageNet')
DNL_PATH = Path(DN_PATH, )
O_PATH = Path(DATA_PATH, 'DAmageNet')
