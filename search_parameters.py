class WindowSearchParameters:
    def __init__(self):
        self.window_size = 96
        self.scale = 1
        self.cell_per_step = 1

        self.y_start_stop = [380, None] #[400, 656]   # Min and max in y to search in slide_window()
