import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.spatial import KDTree
import spacy
from spacy import displacy

def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()

# path is relative to root dir
DATA_PATH = "./data/processed/wellcome1_entitycols.pkl"
DESC_HTML_WRAPPER = """<div style="margin-left: 30px;">{}</div>"""

@st.cache()
def desc_html_wrapper(desc):
    return DESC_HTML_WRAPPER.format(desc)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(data_path):
    df = pd.read_pickle(data_path)
    return df

# functions for NN
def generate_title_tree(df):
    return KDTree(df['title_emb'].tolist())

def get_nearest_neighbours(df, tree, id_, k=5):
    nn_inds = tree.query(df.loc[id_, 'title_emb'][0], k+1)
    query_res = df.iloc[nn_inds[1][1:]][['title', 'category', 'links.self']].copy()
    query_res['distance'] = nn_inds[0][1:]
    query_res = query_res[['title', 'category', 'distance', 'links.self']].rename(columns={'links.self': 'link'})
    return query_res

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    return f'<a target="_blank" href="{link}">{link}</a>'

# functions for NER
def get_NER_model():
    nlp = spacy.load("en_core_web_sm")
    return nlp

@st.cache()
def get_entities(df):
    return [col[4:] for col in df.columns if col.startswith('ent')]

def search_entity(entity, df):
    ids = df[~df['ent_' + entity].isna()].index.tolist()
    
    return ids

def search_entities(entity_list, df):
    if len(entity_list) == 1:
        return search_entity(entity_list[0], df)
    else:
        search_res = [set(search_entity(entity, df)) for entity in entity_list]
        return search_res[0].intersection(*search_res[1:])

welcome_message = """<p style='color:#E60060; font-weight:600; font-size:{}em'>
This is a demo showing how machine learning can be applied to the Science Museum collection to extract meaning from
 text, as part of the Heritage Connector project.
</p>
<p style='color:#000000; font-weight:600; font-size:1.3em'>
by <a style='color:#004899' href='mailto:kalyan.dutia@gmail.com'>Kalyan Dutia</a>
</p>
</br>
"""

def main():
    with st.spinner('loading data..'):
        df = load_data(DATA_PATH)
        
    intro_md = st.markdown(welcome_message.format(1.8), unsafe_allow_html=True)

    page = st.selectbox("Select a page", ["Mapping Collection Items with Word Embeddings", "Creating Structure with Entities"])
    st.markdown("---")
    
    if page == "Mapping Collection Items with Word Embeddings":
        run_page_embeddings(df)

    if page == "Creating Structure with Entities":
        run_page_entities(df)


def run_page_welcome(df):
    pass

def run_page_entities(df):
    st.header("Creating Structure with Entities")
    st.markdown("  ")
    st.markdown(
        "One of the outputs for the Heritage Connector project is as follows:"
    )
    st.markdown(
        "> *'To apply a series of digital tools / computational methods to create speculative identifications between different records within the test dataset.'* "
    )
    st.markdown(
        "Creating identifications in the current collection data is difficult because it is *unstructured*: it consists of mostly text and image fields, which"
        " are hard to reconcile with each other. Here we explore **Named Entity Recognition (NER)** to create some structure from the description field of each"
        " collection document. "
    )
    st.markdown("---")
    
    st.subheader("Enriching Descriptions with NER")
    st.markdown("Named entity recognition extracts known fields of [various types](https://spacy.io/api/annotation#named-entities), using a neural net"
                " and word embeddings (see other page). **Select a collection item below and toggle NER to see it in action.**"
                )
    title = st.selectbox("", df['title'].tolist(), index=16)
    id_ = df[df['title'] == title].index.values
    nlp = get_NER_model()
    desc = df.loc[id_, 'description'].values[0]
    html = displacy.render(nlp(desc), style='ent', minify=True)
    html = html.replace("\n", " ")
    disp_ner = st.checkbox("display entities", value=True)
    st.markdown("  ")
    st.markdown("**Description:**")
    if disp_ner:
        st.markdown(desc_html_wrapper(html), unsafe_allow_html=True)
    else:
        st.markdown(desc_html_wrapper(desc), unsafe_allow_html=True)
    st.markdown("  ")
    st.markdown(f"**Link: ** [{df.loc[id_, 'links.self'].values[0]}]({df.loc[id_, 'links.self'].values[0]})")

    st.markdown("---")

    st.subheader("Towards a Structured Collection")
    st.markdown(
        "Entities can be used to link records together - below is a network of the links between items in different categories using entity"
        " extraction, none of which existed before. The darker the line, the stronger the link. **Toggle the checkbox below to see how entities"
        " have created structure between collection items.**"
    )
    show_graph = st.checkbox('show/hide graph')
    if show_graph:
        st.image("./data/figures/category_network.png", width=600, use_column_width=True)
    
    st.markdown("---")

    st.subheader("Using Entities to Search")
    st.markdown(
        "Once we've created this structure we can use it to search through the collections. This becomes more powerful when we can perform"
        " *Named Entity Linking* to link documents up with an external knowledge base. **Search through the collection using entities below.**"
    )
    st.markdown(
        "I've prefilled Greek items from the 1800s to get you started."
    )
    all_entities = get_entities(df)
    entities = st.multiselect("", all_entities, default=['Greek-NORP', 'the 1800s-DATE'])
    if len(entities) > 0:
        res_ids = search_entities(entities, df)
        if len(res_ids) > 0:
            res_df = df.loc[res_ids, ['title', 'category', 'links.self']]
            res_df = res_df.rename(columns={'links.self': 'link'})
            res_df['link'] = res_df['link'].apply(make_clickable)
            st.write(res_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write(":exclamation:No results returned:exclamation: Change or widen your search.")
    else:
        st.write("Enter some entities in the box above to see some results here :smile:")


def run_page_embeddings(df):
    st.header('Mapping Collection Items with Word Embeddigs')
    st.markdown("  ")
    st.markdown(
        "Word embeddings (e.g. [word2vec](https://jalammar.github.io/illustrated-word2vec)) are a common machine learning method used"
        " to transform words and by extension sentences or paragraphs into fixed-length vectors, where semantically similar words and phrases"
        " have vectors that are closer together."
    )
    st.image("https://www.tensorflow.org/images/linear-relationships.png", caption="source: https://www.tensorflow.org/images/linear-relationships.png", use_column_width=True)
    st.markdown(
        "Here we'll demonstrate two potential uses of word embeddings for Museum collections: identifying groups of similar items using only their"
        " title, and showing a 'map' of collection items."
    )

    st.markdown("---")
    st.subheader('Creating a Map')
    st.markdown(
        "By visualising a two-dimensional representation of word embeddings created from the title of each collection item, we can create a "
        " 'map' of items coloured by Category, as below. A map like this could be tried as a new way of exploring the collection, and also gives us"
        " clues about we could best exploit these embeddings!"
    )
    st.markdown(
        "** Hover over points on the map below to find clusters of similar items.** See if you can find the section containing bottles, or the section containing"
        " stethoscopes, thermometers and defribillators."
    )
    fig = px.scatter(df, x='title_X0', y='title_X1', color='category', hover_data=['title'], width=900, height=600)
    #fig.update_layout({'legend_orientation':'h'})
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig)
    st.markdown("---")
    st.subheader('Searching for Similar Collection Items')
    st.markdown(
        "In this section we'll exploit the property of word embedding vectors that *similar phrases have vectors that are closer together*."
        " The similar item search below works by looking up the vector of the item you choose, and finding the closest items to it (you can"
        " change the number of closest items with the slider)."
    )
    st.markdown(
        "**Pick an item from the collection, then choose the number of similar collection items you want to return.** A similar approach"
        " could be used to fetch similar journal articles to an item, based on their abstracts."
    )
    st.markdown("#### 1. Pick an item")
    title = st.selectbox("", df['title'].tolist(), index=1265)
    id_ = df[df['title'] == title].index.values
    st.markdown(f"**Description: ** {df.loc[id_, 'description'].values[0]}")
    st.markdown(f"**Category: ** {df.loc[id_, 'category'].values[0]}")
    st.markdown(f"**Link: ** [{df.loc[id_, 'links.self'].values[0]}]({df.loc[id_, 'links.self'].values[0]})")

    st.markdown("#### 2. Find similar items")
    k = st.slider("Number of similar items to display", min_value=1, max_value=12, value=5)
    tree = generate_title_tree(df)
    nns = get_nearest_neighbours(df, tree, id_, k)
    #st.table(nns)
    nns['link'] = nns['link'].apply(make_clickable)
    st.write(nns.to_html(escape=False, index=False), unsafe_allow_html=True)

#
#nlp = get_NER_model()
#html = displacy.render(nlp(df.loc[id_, 'description'].values[0]), style='ent', minify=True)
#html = html.replace("\n", " ")
#st.markdown(DESC_HTML_WRAPPER.format(html), unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()