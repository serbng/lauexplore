def grid_chunks(nbrows: int, nbcols: int, chunksize: int):
    """Given a grid of shape (num_rows, num_cols), return the linear indices that subdivide it
    in square chunks of shape (chunksize, chunksize).
    

    Args:
        num_rows  (int): Number of rows of the grid
        num_cols  (int): Number of columns of the grid
        chunksize (int): Size of the side of the square chunk

    Returns:
        chunks (list[list[int]]): List whose elements are a list of the linear indices
                                  corresponding to the same sub-grid/chunk.
        
    Example:
    >>> import matplotlib.pyplot as plt
    >>> from graintools import utils
    >>> scan_points = utils.mesh_points((0,0), (130,50), (1, 1))
    >>> scan_points.shape
        (6681, 2)
    >>> chunks = utils.grid_chunks(51, 131, 20)
    >>> fig, ax = plt.subplots()
    >>> for chunk in chunks:
    ...     ax.scatter(scan_points[chunk, 0], scan_points[chunk, 1])
    >>> ax.set_aspect('equal')
    """
    chunks = []
    for r_start in range(0, nbrows, chunksize):
        r_end = min(r_start + chunksize, nbrows)
        for c_start in range(0, nbcols, chunksize):
            c_end = min(c_start + chunksize, nbcols)
            
            # Costruiamo la lista di indici "lineari" di questo sotto-blocco
            subgrid_indices = []
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    idx = r * nbcols + c
                    subgrid_indices.append(idx)
            
            chunks.append(subgrid_indices)
    
    return chunks

def linear_chunks(length: int, chunksize: int):
    """Given the length of a list, return the linear indices that subdivide it
    in sub-lists of length chunksize.
    

    Args:
        length    (int): Length of the list to subdivide
        chunksize (int): Length of each sub-list

    Returns:
        chunks (list[list[int]]): List whose elements are a list of the linear indices
                                  corresponding to the same sub-list/chunk.
        
    Example:
    >>> import matplotlib.pyplot as plt
    >>> from graintools import utils
    >>> scan_points = utils.mesh_points((0,0), (130,50), (1, 1))
    >>> scan_points.shape
        (6681, 2)
    >>> chunks = utils.grid_chunks(6681, 500)
    >>> fig, ax = plt.subplots()
    >>> for chunk in chunks:
    ...     ax.scatter(scan_points[chunk, 0], scan_points[chunk, 1])
    >>> ax.set_aspect('equal')
    """
    indices = list(range(length))
    chunks = []
    for start in range(0, length, chunksize):
        end = min(start + chunksize, length)
        chunk = indices[start:end]
        chunks.append(chunk)
        
    return chunks