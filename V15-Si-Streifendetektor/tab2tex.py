#  import uncertainties


def make_table(
        header,
        places,
        data,
        caption = 'tab2tex-generated.',
        label = 'tab:tab2tex-generated',
        filename = 'tab2tex-generated.tex'):
    '''Method to generate table in build folder'''
    # review input
    if not type(caption) == str:
        raise TypeError('Caption has to be string')
        return
    if not type(label) == str:
        raise TypeError('Label has to be string')
        return
    if not len(places) == len(data):
        raise IndexError('places and data have to have the same length')
        return
    if len(data) == 0:
        raise IndexError('No data given')
        return
    for param in range(0, len(data)):
        if not len(data[param]) == len(data[0]):
            print('Warning: Data columns of different length')
        if len(data[param]) == 0:
            raise IndexError(f'Data column {param} is empty')
    # TODO: Check, if places is a tuple whenever data has an uncertainty object

    # Try to open file
    try:
        file = open(filename, 'w')  # Overwrite when existing
    except IOError:
        print('Error: Can\'t write given filename')
        return

    # Header
    file.write('\\begin{table}' + '\n')
    file.write('\t' + '\\centering' + '\n')
    file.write('\t' + '\\caption{' + caption + '}' + '\n')
    file.write('\t' + '\\label{' + label + '}' + '\n')
    file.write('\t' + '\\begin{tabular}{' + '\n')
    # recognize columns with text an uncertainty arrays
    for param in range(0, len(places)):
        if type(places[param]) == tuple:
            file.write('\t\t' + 'S[table-format=' + str(places[param][0]))
            file.write('] @{${}\pm{}$} S[table-format=' + str(places[param][1]))
            file.write(']' + '\n')
        elif type(data[param][0]) == str:
            file.write('\t\t' + 'c' + '\n')
        else:
            file.write('\t\t' + 'S[table-format=' + str(places[param]) + ']' + '\n')
    file.write('\t\t' + '}' + '\n')

    # Toprule
    file.write('\t' + '\\toprule' + '\n')
    # recognition, if parameter has a unit with expression ' / ' in header
    for head in range(0, len(header)):
        last_head = head == len(header)-1
        unit = False
        if ' / ' in header[head]: unit = True
        if type(places[head]) == tuple:
            file.write('\t\t' + '\multicolumn{2}{c}{')
        else:
            file.write('\t\t' + '{')
        if unit:
            file.write(header[head].split(' / ')[0])
            file.write('\\;/\\;\\si{' + header[head].split(' / ')[1] + '}')
        else:
            file.write(header[head])
        file.write('} \\\\' + '\n') if last_head else file.write('} &' + '\n')

    # Midrule
    file.write('\t' + '\\midrule' + '\n')
    # columns can have different length
    for row in range(0, len(max(data, key=len))):
        file.write('\t\t')
        for param in range(0, len(data)):
            if row >= len(data[param]):  # prevent index out of bound
                if type(places[param]) == tuple: file.write(' & ')
            else:
                if type(places[param]) == tuple:  # Column with uncertainty array
                    file.write('{: {}f}'.format(data[param][row].n, places[param][0]))
                    file.write(' & ' + '{: {}f}'.format(data[param][row].s, places[param][1]))
                elif type(data[param][0]) == str:  # Column with strings
                    file.write(data[param][row])
                else:
                    file.write('{: {}f}'.format(data[param][row], places[param]))
            file.write(' \\\\' + '\n') if param == len(data)-1 else file.write(' & ')

    # Bottomrule
    file.write('\t' + '\\bottomrule' + '\n')
    file.write('\t' + '\\end{tabular}' + '\n')
    file.write('\\end{table}')

    file.close()
    print(filename + ' written')
    return


#  def fromUncertainties(variable):
    #  '''Tests if variable is an uncertainties object'''
    #  if type(variable) == uncertainties.core.Variable:
        #  return True
    #  if type(variable) == uncertainties.core.AffineScalarFunc:
        #  return True
    #  return False

