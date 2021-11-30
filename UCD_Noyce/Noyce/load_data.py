
import pandas as pd
from utils.normalizer import normalize
from sklearn.model_selection import train_test_split

FACEBOOK_POSTS = "./UCD_Noyce/Noyce/data/ideology/facebook.csv"
YOUTUBE_POSTS = "./UCD_Noyce/Noyce/data/ideology/youtube.csv"
REDDIT_COMMENTS = "./UCD_Noyce/Noyce/data/ideology/reddit_comments_onesided.csv"
REDDIT_COMMENTS_POL = "./UCD_Noyce/Noyce/data/ideology/reddit_political_comments_85.csv"

def load_filterer():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/FILTERER/train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/FILTERER/test.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_FIN():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_FIN.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_FIN.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()

def load_ideo_addn_yt_slant_pol2():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_yt_slant_pol2.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old2.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df_test = df_test.loc[df_test['class_id'] != 2]
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_yt_slant_pol():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_yt_slant_pol.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df_test = df_test.loc[df_test['class_id'] != 2]
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_yt_slant_combined2():

    df1 = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_yt_slant_cleaned.csv", encoding='unicode_escape').dropna()
    df2 = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_new.csv", encoding='unicode_escape').dropna()
    
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape').dropna()
    
    df = pd.concat([df1, df2], join='outer', ignore_index=False)
    df_test = df_test.loc[df_test['class_id'] != 2]
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_yt_slant_only():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_yt_slant_cleaned.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df_test = df_test.loc[df_test['class_id'] != 2]
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_yt_slant_combined():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_yt_slant_combined.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_slant_filtered():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_slant_filtered.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df = df.loc[df['class_id'] != 2]
    df_test = df_test.loc[df_test['class_id'] != 2]
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()

def load_ideo_addn_slant1():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_slant.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_slant2():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_new_slant.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_newold():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_new.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    
    df = df.loc[df['class_id'] != 2]
    df_test = df_test.loc[df_test['class_id'] != 2]
    
    df = df.sample(frac=1, random_state=42)
    df_test = df_test.sample(frac=1, random_state=42)
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_final():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/combined/train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/combined/test.csv", encoding='unicode_escape')
    
    df = df.dropna()
    df_test = df_test.dropna()
    df = df.sample(frac=1)
    df_test = df_test.sample(frac=1)
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()

def load_ideo_addn_bin():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/combined/train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/combined/test.csv", encoding='unicode_escape')
    
    df = df.dropna()
    
    df = df.loc[df['class_id'] != 2]
    df_test = df_test.loc[df_test['class_id'] != 2]
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()

def load_ideo_addn_ninety():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_ninety.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_ten.csv", encoding='unicode_escape')
    
    df = df.dropna()
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_addn_old():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/train_with_old.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/IDEOLOGICAL/additionals/test_with_old.csv", encoding='unicode_escape')
    
    df = df.dropna()
    
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].astype(int).tolist(), df_test['text'].tolist(), df_test['class_id'].astype(int).tolist()


def load_ideo_speech():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/anshuman_ideology/ideology_train_speech.csv", encoding='utf-8')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/anshuman_ideology/ideology_test.csv", encoding='utf-8')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_ideo_v1():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/anshuman_ideology/ideology_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/anshuman_ideology/ideology_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_pol_final():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/final_version/combined/train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/final_version/combined/test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_pol_v1():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/v1/political_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/v1/political_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()

def load_pol_v2():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/v2/political_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/v2/political_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()

def load_pol_v3():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/v3/political_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/POLITICAL/v3/political_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_new_ideo_data():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/ideology/ideology_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/ideology/ideology_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_pol_data():

    df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/anshuman_political/political_train.csv", encoding='unicode_escape')
    df_test = pd.read_csv(
        "./UCD_Noyce/Noyce/data/anshuman_political/political_test.csv", encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    df_test['text'] = df_test['text'].apply(normalize)
    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()

def load_csv(path):
    df = pd.read_csv(path, encoding='unicode_escape')
    df['text'] = df['text'].apply(normalize)
    return df['text'].tolist()

def load_ideology_data(website, separate_websites = False, test_set = True):
    test_size = 0.1
    if (website == 'facebook'):
        path = FACEBOOK_POSTS
        test_size = 0.025
        df = pd.read_csv(path, encoding='unicode_escape')

    elif (website == 'youtube'):
        path = YOUTUBE_POSTS
        df = pd.read_csv(path, encoding='unicode_escape')

    elif (website == 'redditcomments'):
        path = REDDIT_COMMENTS
        df = pd.read_csv(path, encoding='unicode_escape')
        test_size = 0.01


    elif (website == 'youtube_facebook'):
        df1 = pd.read_csv(YOUTUBE_POSTS, encoding='unicode_escape')
        df2 = pd.read_csv(FACEBOOK_POSTS, encoding='unicode_escape')
        df = pd.concat([df1,df2])
        test_size = 0.020

    elif (website == 'all'):
        df1 = pd.read_csv(YOUTUBE_POSTS, encoding='unicode_escape')
        df2 = pd.read_csv(FACEBOOK_POSTS, encoding='unicode_escape')
        df3 = pd.read_csv(REDDIT_COMMENTS, encoding='unicode_escape')
        df = pd.concat([df1,df2,df3])
        df = df.sample(frac = 1, random_state = 30)
        test_size = 0.01


    elif (website == 'redditcomments_pol'):
        path = REDDIT_COMMENTS_POL
        df = pd.read_csv(path)
        df = df.sample(frac = 1, random_state = 30)
        test_size = 0.05
    else:
        df = pd.read_csv(website, encoding='unicode_escape')



    df['text'] = df['text'].apply(normalize)
    df = df.dropna()
    
    if separate_websites:
        df_test =  pd.concat([df[(df['website'] == 'colorlines')], df[df['website'] == 'mrc' ]]) 
        df = df[(df['website'] != 'colorlines')]
        df = df[(df['website'] != 'mrc')]

    else:
        if test_set:
            df ,df_test = train_test_split(df, random_state=1, test_size=test_size, stratify = df['class_id'])
        else:
            return  df['text'].tolist(), df['class_id'].tolist(), None, None

    return df['text'].tolist(), df['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()

def load_disagreement_data():
    class_id_dict = {
        "SE" : 0,
        "AC" : 0,
        "AE" : 0,
        "DE" : 1,
        "DC" : 1,
        "DC/AC" : 1,
        "DE/DC" : 1,

    }

    YT_df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/disagreement/Youtube_Disagreement_Comments.csv", encoding='unicode_escape')[['text','class']]

    FB_df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/disagreement/Facebook_Disagreement_Comments.csv", encoding='unicode_escape')[['text','class']]

    Reddit_df = pd.read_csv(
        "./UCD_Noyce/Noyce/data/disagreement/Reddit_Disagreement_Comments.csv", encoding='unicode_escape')[['text','class']]

    df = pd.concat([YT_df, FB_df, Reddit_df])
    df['class_id'] = df['class'].map(class_id_dict)
    df = df.dropna()
    df_train ,df_test = train_test_split(df, random_state=1, test_size=0.1, stratify = df['class_id'])
    df_train.loc[:,'text'] = df_train.text.apply(normalize)
    df_test.loc[:,'text'] = df_test.text.apply(normalize)

    return df_train['text'].tolist(), df_train['class_id'].tolist(), df_test['text'].tolist(), df_test['class_id'].tolist()


def load_data(dset_name='political_final', path = '', test_set = True):
    if dset_name == 'filterer':
        return load_filterer()
    if dset_name == 'ideology_FIN':
        return load_ideo_FIN()
    if dset_name == 'ideology_addn_yt_slant_pol2':
        return load_ideo_addn_yt_slant_pol2()
    if dset_name == 'ideology_addn_yt_slant_pol':
        return load_ideo_addn_yt_slant_pol()
    if dset_name == 'ideology_addn_yt_slant_combined2':
        return load_ideo_addn_yt_slant_combined2()
    if dset_name == 'ideology_addn_yt_slant_only':
        return load_ideo_addn_yt_slant_only()
    if dset_name == 'ideology_addn_yt_slant_combined':
        return load_ideo_addn_yt_slant_combined()    
    if dset_name == 'ideology_addn_slant_filtered':
        return load_ideo_addn_slant_filtered()
    if dset_name == 'ideology_addn_newslant':
        return load_ideo_addn_slant2()
    if dset_name == 'ideology_addn_slant':
        return load_ideo_addn_slant1()
    if dset_name == 'ideology_addn_newold':
        return load_ideo_addn_newold()
    if dset_name == 'ideology_final':
        return load_ideo_final()
    if dset_name == 'ideology_addn_bin':
        return load_ideo_addn_bin()
    if dset_name == 'ideology_addn_ninety':
        return load_ideo_addn_ninety()
    if dset_name == 'ideology_addn_old':
        return load_ideo_addn_old()
    if dset_name == 'ideology_v1':
        return load_ideo_v1()
    if dset_name == 'ideology_speech':
        return load_ideo_speech()
    if dset_name == 'political_final':
        return load_pol_final()
    if dset_name == 'political_v1':
        return load_pol_v1()
    if dset_name == 'political_v2':
        return load_pol_v2()
    if dset_name == 'political_v3':
        return load_pol_v3()
    if dset_name == 'political_text':
        return load_pol_data()
    if dset_name == 'ideology':
        return load_new_ideo_data()
    elif dset_name == 'disagreement':
        return load_disagreement_data()
    elif dset_name == 'ideology_fb':
        return load_ideology_data('facebook',test_set = test_set)
    elif dset_name == 'ideology_youtube':
        return load_ideology_data('youtube', test_set =test_set)
    elif dset_name == 'ideology_redditcomments':
        return load_ideology_data('redditcomments', test_set =test_set)
    elif dset_name == 'ideology_redditcomments_pol':
        return load_ideology_data('redditcomments_pol', test_set =test_set)
    elif dset_name == 'ideology_facebook_youtube':
        return load_ideology_data('youtube_facebook', test_set =test_set)
    elif dset_name == 'ideology_all':
        return load_ideology_data('all', test_set =test_set)
    elif dset_name == 'ideology_custome':
        return load_ideology_data(path, test_set)

    else:
        raise NameError(
            'Dataset not known. Available Datasets: political_text')

if __name__ == '__main__':
    print(len(load_data(dset_name = 'ideology_redditcomments_pol_balanced')[1]),len(load_data(dset_name = 'ideology_youtube')[0]),
    len(load_data(dset_name = 'ideology_youtube')[2]),len(load_data(dset_name = 'ideology_youtube')[3]))
