


def print_tree_structure(tree: object, properties: list = None) -> None:
    '''
    Function to print out the structure of a tree classifier, can also print values for each node.

    :param tree: Tree class object which has been trained. 
    :param properties: Given properties to print for each node. 

    :returns: None 
    '''
    type_names = ['Root', 'branch', 'leaf']

    if properties is  None:
        property_list = ''
    else:
        property_list = []
        for prop in properties:
            try:
                property_list.append(eval(f'tree.{prop}'))
            except AttributeError:
                property_list.append('')
    

    
    print('   ' * tree.depth + type_names[tree.type] + ' node  ', property_list) 

    if tree.type != 2:

        for child in tree.children:
            
            print_tree_structure(child, properties)

    return None    




if __name__ == '__main__':
    pass