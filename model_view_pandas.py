from PyQt5 import QtCore, QtGui, QtWidgets


class pandasModel(QtCore.QAbstractTableModel):
    def __init__(self,data):
        QtCore.QAbstractTableModel.__init__(self)
        self.data = data

    def rowCount(self, parent=None):
        return self.data.shape[0]

    def columnCount(self, parent=None):
        return self.data.shape[1]

    def data(self, index,role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self.data.iloc[index.row(), index.column()])
            return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.data.columns[col]
        return None
