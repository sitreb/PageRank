import numpy as np
import pandas as pd
import random
from scipy import sparse

""" Function to convert a list of edges in the form of a pandas dataframe into a column stochastic sparse adjacency
    matrix to be used in the pagerank algorithm """

def get_stochastic_adj_matrix(df):

    # df is a Pandas dataframe

    # Extract from the dataframe all the page titles

    vals = np.unique(df[['page_title_from', 'page_title_to']])
    n = len(vals)

    """ Create a dataframe with two columns: one containing the page titles (alphabetically ordered) 
        and the other containing the new ids we are giving them """

    data = pd.DataFrame(vals, columns=['vals'])
    data['id'] = range(n)

    """ Consider only the two columns with the page titles in the original dataframe
        and merge twice to obtain the ids of the pages from which a link starts and the ids of the pages to which
        a link points. We will use them as coordinates to create the sparse adjacency matrix """

    df = df[['page_title_from', 'page_title_to']]
    new_df = pd.merge(df, data, left_on='page_title_from', right_on='vals').drop('vals', axis=1)
    new_df = pd.merge(new_df, data, left_on='page_title_to', right_on='vals')\
            .rename({'id_x':'id_from', 'id_y':'id_to'}, axis=1).drop('vals', axis=1)

    # Create a dataframe that associates to each page from which a link starts the number of its out-links and merge

    df_out = df.groupby('page_title_from', as_index=False).count().rename({'page_title_to':'n_outlinks'}, axis=1)
    new_df = pd.merge(new_df, df_out, left_on='page_title_from', right_on='page_title_from')

    """ Construct the sparse matrix in coordinate format, which is a fast format for constructing this type of matrices.
        The coordinates are the new ids we have constructed and the data is given by 1/n_outlinks """

    row = np.array(new_df['id_to'])
    col = np.array(new_df['id_from'])
    data = np.divide(1, np.array(new_df['n_outlinks']))

    M = sparse.coo_matrix((data, (row,col)), shape=(n,n))

    """ Convert the coordinate format sparse matrix into a list of list sparse matrix, which is more convenient for
        constructing sparse matrices incrementally since it can be sliced.
        Consider the columns that do not sum up to one (i.e. the columuns that sum up to zero since they are dead ends:
        they have no out-links) and fill the rows with 1/n.
        This makes the matrix column stochastic """

    M = M.tolil()

    sums = M.sum(axis=0)
    i,j = sums.nonzero()
    k = np.arange(0,n)
    mask = np.zeros(n, dtype=bool)
    mask[j] = True
    col_zero_sum = k[~mask]

    M[:, col_zero_sum] = 1/n

    # Convert to compressed sparse row matrix format, which is more efficient for arithmetic operations
    
    M = M.tocsr()

    return M

######################### "Naive" Pagerank #########################

""" The "naive" pagerank algorithm does not take into account spider traps

    Here, the algorithm is just defined but it is not run with test data since it is very inefficient: I have tried 
    running it and 1000 iterations were not enough for convergence """

def NaivePageRank(M):
    
    if np.all(abs(M.sum(axis=0) - 1) > 1.0e-9):
        raise Exception("The matrix is not column stochastic")
    
    counter = 0
    n = M.shape[0]
    norm = 1
    r = np.ones((n, 1))/n
    r_updated = r
    
    while norm > 1.0e-6:
        r = r_updated
        r_updated = M @ r
        norm = np.linalg.norm(r_updated - r, 1)
        counter += 1
        
    print(f"Number of iterations for convergence: {counter}")

    return np.array(r_updated)

######################### Pagerank #########################

def PageRank(M, beta=0.85):

    # M is a column stochastic sparse matrix
    # beta is the dumping factor and it defaults to 0.85
    
    """ Check that the matrix M is column stochastic by checking that the difference in absolute value between the sum
        of each column and 1 is less than an arbitrarily small value. This takes into consideration the fact that there
        might be some sort of approximation and so that the sum of a column might not be exactly one, but it still
        needs to be very close to one """

    if np.all(abs(M.sum(axis=0) - 1) > 1.0e-9):
        raise Exception("The matrix is not column stochastic")

    if beta <= 0 or beta >= 1:
        raise Exception("beta must be in (0, 1)")
    
    counter = 0
    n = M.shape[0]
    norm = 1
    r = np.ones((n, 1))/n
    r_updated = r
    
    G = beta * M + (1 - beta) * np.ones((n, n))/n

    print("Computing PageRank", end=" ")
    
    while norm > 1.0e-6:
        r = r_updated
        r_updated = G @ r
        norm = np.linalg.norm(r_updated - r, 1)
        counter += 1
        print(".", end="")
        
    print(f"\nNumber of iterations for convergence: {counter}")
    
    return np.array(r_updated)

######################### Personalized Pagerank #########################

def PersonalizedPageRank(M, p, a=0.8, b=0.15, c=0.05):

    # p is the personalization vector specific to a given user

    if np.all(abs(M.sum(axis=0) - 1) > 1.0e-9):
        raise Exception("The matrix is not column stochastic")

    if abs(p.sum(axis=0) - 1) > 1.0e-9:
        raise Exception("The entries of the personalization vector do not sum up to 1")

    if abs(a + b + c - 1) > 1.0e-9:
        raise Exception("a, b and c do not sum up to 1")

    if a <= 0 or b <= 0 or c <= 0:
        raise Exception("a, b and c must be positive")

    counter = 0
    n = M.shape[0]
    norm = 1
    r = np.ones((n, 1))/n
    r_updated = r

    print("Computing Personalized PageRank", end=" ")
    
    while norm > 1.0e-6:
        r = r_updated
        r_updated = a * M @ r + b * p + c * np.ones((n,1))/n
        norm = np.linalg.norm(r_updated - r, 1)
        counter += 1
        print(".", end="")
        
    print(f"\nNumber of iterations for convergence: {counter}")
    
    return np.array(r_updated)

                         #########################

# Function to get a random personalization vector of size n, with m non-zero entries

def get_random_personalization_vector(n, m):

    """ args: n = lenght of the personalization vector
              m = number of entries that have a non-zero probability (corresponding to pages to which a given user has
              a positive probability to teleport) """

    personalization_vector = sparse.lil_matrix((n, 1))
    rand = np.random.rand(int(m), 1)
    rand = np.divide(rand, np.sum(rand))
    rand_indexes = np.array(random.sample(range(0, n), int(m)))
    personalization_vector[rand_indexes, 0] = rand
    return personalization_vector.tocsr()

######################### Random Walk with Restart #########################

def RandomWalkWithRestart(M, S, a=0.8, b=0.15, c=0.05):

    # S is the id of the page from which the random walk starts

    if np.all(abs(M.sum(axis=0) - 1) > 1.0e-9):
        raise Exception("The matrix is not column stochastic")

    if abs(a + b + c - 1) > 1.0e-9:
        raise Exception("a, b and c do not sum up to 1")

    if a <= 0 or b <= 0 or c <= 0:
        raise Exception("a, b and c must be positive")

    counter = 0
    n = M.shape[0]
    norm = 1
    r = np.ones((n, 1))/n
    r_updated = r

    p = sparse.lil_matrix((n, 1))
    p[S, 0] = 1
    p = p.tocsr()

    print("Computing Random Walk with Restart", end=" ")
    
    while norm > 1.0e-6:
        r = r_updated
        r_updated = a * M @ r + b * p + c * np.ones((n,1))/n
        norm = np.linalg.norm(r_updated - r, 1)
        counter += 1
        print(".", end="")
        
    print(f"\nNumber of iterations for convergence: {counter}")
    
    return np.array(r_updated)

                         #########################

""" Function to get a dictionary of the Wikipedia pages sorted by their rank value obtained through
    one of the previous algorithms """

def get_sorted_rank(vals, r):

    """ args: vals = numpy array containing the page titles in alphabetical order
              r = numpy array containing the ranks

        They both have the same ordering, i.e. r[n] is the rank of vals[n], since we have created the
        stochastic adjacency matrix using ids corresponding to the alphabetically ordered page titles """

    d = dict(zip(vals, r[:,0]))
    sorted_dict = {}
    sorted_keys = sorted(d, key=d.get, reverse=True)
    for i in sorted_keys:
        sorted_dict[i] = d[i]
    return sorted_dict

                         #########################

# Function to get the first n items from a sorted dictionary

def get_first_n_items_from_dict(d, n):

    """ args: d = sorted dictionary
              n = (int) number of items we want """

    if not isinstance(n, int):
        raise Exception("n must be an integer value")

    dict_items = d.items()
    first_n = list(dict_items)[:n]
    return first_n

####################################################################################################

""" The dataset I have is used can be dowloaded from https://zenodo.org/record/2539424#.YW2aHtlBz8H and it is the one
    called 'enwiki.wikilink_graph.2002-03-01.csv'. It should be placed in the same directory of this file.
    WikiLinkGraphs is a dataset of the network of internal Wikipedia links for 9 language editions and it spans over 17 years,
    from the creation of Wikipedia in 2001 to March 2018.
    The original paper is: Consonni Cristian, Laniado David, and Montresor Alberto. “WikiLinkGraphs: A complete, longitudinal
    and multi-language dataset of the Wikipedia link networks.”

    I decided to use the dataset corresponding to the year 2002 for the English language, since it is big enough to make 
    computation challenging but it can still be run in a minute or so. Bigger datasets would probably require hours to run
    and the Python languge would probably not be adequate anymore. """

df = pd.read_csv('enwiki.wikilink_graph.2002-03-01.csv', sep='\t')
vals = np.unique(df[['page_title_from', 'page_title_to']])
n = len(vals)

M = get_stochastic_adj_matrix(df)

PR_ranks = PageRank(M)

d = get_sorted_rank(vals, PR_ranks)
first_ten = get_first_n_items_from_dict(d, 10)
print(f"The first ten Wikipedia pages ordered using PageRank and the corresponding scores are the following:\n{first_ten}")

print("\nLet us consider 3 different users, each with his own preferences, and compute for each of them the\
 Personalized PageRank algorithm (the preferences are randomly computed):")

# Suppose each of them is interested in 1000 random pages

p = [get_random_personalization_vector(n, 1000) for i in range(3)]

PPR_ranks_user = [PersonalizedPageRank(M, p[i]) for i in range(3)]

for i in range(3):
    d = get_sorted_rank(vals, PPR_ranks_user[i])
    first_ten = get_first_n_items_from_dict(d, 10)
    print(f"The first ten Wikipedia pages ordered using Personalized PageRank and the corresponding scores for user {i+1}\
 are the following:\n{first_ten}")

data = pd.DataFrame(vals, columns=['vals'])
data = data.to_dict()

S = [np.random.randint(n) for i in range(3)]

print(f"\nLet us compute 3 Random Walk with Restart, starting from the Wikipedia pages '{data['vals'][S[0]]}',\
 '{data['vals'][S[1]]}' and '{data['vals'][S[2]]}'")

RWS_ranks = [RandomWalkWithRestart(M, S[i]) for i in range(3)]

for i in range(3):
    d = get_sorted_rank(vals, RWS_ranks[i])
    first_ten = get_first_n_items_from_dict(d, 10)
    print(f"The first ten Wikipedia pages ordered using a Random Walk with Restart starting from '{data['vals'][S[i]]}'\
 and the corresponding scores are the following:\n{first_ten}")


