c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.allow_origin = '*'
c.NotebookApp.disable_check_xsrf = True

# Read-only: No delete, no update, no rename, no create
c.ContentsManager.delete_to_trash = False
c.ContentsManager.allow_hidden = False
c.FileContentsManager.delete_to_trash = False

# Disable file modifications
c.NotebookApp.terminals_enabled = False
c.NotebookApp.allow_root = True

# Custom config to block delete/rename/save
c.ContentsManager.checkpoints_kwargs = {'root_dir': '/tmp/checkpoints'}
