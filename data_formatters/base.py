import enum
import abc
class InputType(enum.IntEnum):
    TARGET = 0
    ID = 1
    TIME = 2

class GenericDataFormatter(abc.ABC):

    @property
    @abc.abstractclassmethod
    def _column_definition(self):
        raise NotImplementedError()
    
    def get_column_definition(self):

        column_definition = self._column_definition

        def _check_single_column(input_type):

            length = len([tup for tup in column_definition if tup[1] == input_type])

            if length != 1:
                raise ValueError('Illegal number of inputs({}) of type {}'.format(length, input_type))
        
        _check_single_column(InputType.ID)
        _check_single_column(InputType.TIME)

        indentifier = [tup for tup in column_definition if tup[1] == InputType.ID]
        time = [tup for tup in column_definition if tup[1] == InputType.TIME]
        real_inputs = [tup for tup in column_definition if tup[1] == InputType.TARGET]

        return indentifier + time + real_inputs
    
    @property
    @abc.abstractmethod
    def split_data(self, df):
        raise NotImplementedError
        
