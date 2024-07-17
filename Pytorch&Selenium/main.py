# Selenium Web bileşenlerini ekleyebiliyoruz
# pip install selenium
# Pytorch Makine Öğrenmesi ve Derin Öğrenme (Ekstradan Dil İşleme Var)
# pip install torch

# Web işlemlerine başlayalım

from selenium import webdriver
import time

'''
brw = webdriver.Firefox()

# brw = webdriver.Chrome()

brw.get('https://www.selenium.dev/')

# time modülü
time.sleep(3)

brw.quit() # Kapatmak için

# T3 Akademi resmi sitesine gitme

tarayici = webdriver.Chrome()

time.sleep(1)
tarayici.get('https://www.t3vakfi.org/tr/')

time.sleep(5)
tarayici.quit()

# webde yapay zeka kelimesini aratalım: APjFqb

from selenium.webdriver.common.by import By # Her bir HTML özelliğine erişecek
from selenium.webdriver.common.keys import Keys # Klavye simülasyonu için

brows = webdriver.Chrome()
time.sleep(1)
brows.get('https://www.google.com')
brows.set_window_size(800, 600)
brows.refresh()

time.sleep(1)
search = brows.find_element(By.ID, 'APjFqb') # Arama çubuğunu ID bilgisini göndererek buluyor
search.send_keys('yapay zeka'+Keys.ENTER) # Başka bir şey daha kullanılabilir

time.sleep(3)
brows.quit()
'''

# pytorch kütüphanemizden yapay sinir ağları modeli oluşturalım

import torch

# ysa model oluşturmak için 

import torch.nn as nn

# optimizasyon algoritması için

import torch.optim as optim
import numpy as np

# veri setini dahil edelim

data_set = np.loadtxt('diabetes_data.csv', skiprows=1, delimiter = ',') # ilk satırı atlar ve geri kalan virgüllere göre ayırır

# features-target ayrımı

X = data_set[:,:8]
Y = data_set[:, 8]

# numpy arraylerini pytorch tensörleine dönüştürme
# Çünkü pytorch tensörlerle işlem yapar
# vektörel ve skaler
# tensör derecesi 0 olunca sayı skaler oluyor

'''
tensör 0: skaler
tensör 1: vektörel sayı
tensör 2: matris (Kütle çekim dalgaları) Bilimde en üst nokta (şu anlık)
tensör 3: çok boyutlu veri yapısı (Bilim Kurgu)
# Dalgalarla ilgili bir kitap vardı
'''

X = torch.tensor(X, dtype=torch.float32) # Buradaki verileri azaltabiliyoruz çünkü önemli veriler değil
Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1) # dönüştürme işlemi: -1: 64 => 32 ye geçiş, 1: tensör 1 e geçiş

# ysa eğitim algoritması: Ann_model

import torch.nn as nn # Göstermelik

Ann_model = nn.Sequential(
    nn.Linear(8,16), # Input veriler için
    nn.ReLU(), # 0 ' dan aşağısı yok
    nn.Linear(16, 10), # Gizli katman
    nn.ReLU(),
    nn.Linear(10, 1), # Gizli katman ama çıkış için
    nn.Sigmoid()) # Output (Çıkış)

print('Oluşturulan Model:', Ann_model)

# optimizer algoritması tanımlyoruz: Adam Algoritması

optimizer_alg = optim.Adam(Ann_model.parameters(), lr=0.002)

# Kayıp fonksiyonu

loss_func = nn.BCELoss()

batch_size = 10
total_epoch = 200

# Gradyan ? !!!

# ysa modelimizi eğitelim: Önceden fit vardı

for e in range(total_epoch):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        Y_pred = Ann_model(X_batch)
        Y_batch = Y[i:i+batch_size]
        loss_s = loss_func(Y_pred, Y_batch)
        # Soru çözüp cevaba bakıyor
        optimizer_alg.zero_grad() # Her seferinde optimize etmesi gerek
        loss_s.backward() # Gidişatın nereye gittiğini öğreniriz (Türev-değişim)
        optimizer_alg.step()
        
    print(f'Oluşturulan epoch değeri: {e}, loss kayıp değeri: {loss_s}')

# Model değerlendirme

with torch.no_grad():
    Y_pred = Ann_model(X)
    
# acc score

acc_score = (Y_pred.round() == Y).float().mean()

print('Doğruluk Skoru:', acc_score)



