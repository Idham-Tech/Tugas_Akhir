import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils import *

st.title("Prediksi Deret Waktu Pada Nilai Ekspor Non-Migas di Indonesia")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

@st.cache_data
def load_csv(path: str):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'], format='%Y %B')
    data = data.sort_values('date')
    return data

data = load_csv('./Keseluruhan (Coba-coba) NonMigas.csv')
# st.write(df)

# sidebar #
st.sidebar.write('Data as on [Badan Pusat Statistik](https://www.bps.go.id/id) export value')
option = st.sidebar.selectbox('What would you like to predicted?',('Stacked GRU', 'Bidirectional GRU', 'Attention + GRU', 'STL + GRU','Compare'), key = 'tick')
st.sidebar.write('You selected:', st.session_state.tick)
predict = st.sidebar.button('Predict')
st.sidebar.info('''This Project is used for only learning and development process.😊👩🏻‍🎓👩🏻‍💻''')

try:
    if predict:
            if option == 'Stacked GRU':    
                # show data
                data_act()
                st.write(data)

                # visualizations
                visual_data()
                plot_actual_data(data['date'], data['NonMigas'])

                # Proccess prediction
                proccess(option)

            elif option == 'Bidirectional GRU':

                # show data
                data_act()
                st.write(data)

                # visualizations
                visual_data()
                plot_actual_data(data['date'], data['NonMigas'])

                # Proccess prediction
                proccess(option)

            elif option == 'Attention + GRU':

                # show data
                data_act()
                st.write(data)

                # visualizations
                visual_data()
                plot_actual_data(data['date'], data['NonMigas'])

                # Proccess prediction
                proccess(option)

            elif option == 'STL + GRU':

                # show data
                data_act()
                st.write(data)

                # visualizations
                visual_data()
                plot_actual_data(data['date'], data['NonMigas'])

                # Proccess prediction
                proccess(option)

            else:

                # show data
                data_act()
                st.write(data)

                # visualizations
                visual_data()
                plot_actual_data(data['date'], data['NonMigas'])

                # Proccess prediction
                proccess(option)

    else:
        st.info('Please, choose the available options then click predict button🙏🏻')
except Exception as e: 
    st.error(e)