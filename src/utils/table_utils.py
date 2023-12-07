import wandb

class TableUtils():
    def __init__(self, columns, name):
        self.columns = columns
        self.name = name
        # self.table = wandb.Table(columns=columns)
        self.data = []
    
    def add_row(self, row):
        # self.table.add_data(*row)
        self.data.append(row)

    def return_data(self):
        data = self.data
        self.data = []
        return data
        
    def return_table(self):
        return_table = self.table
        # reset the table
        self.table = wandb.Table(columns=self.columns)
        return return_table # how the turntables
