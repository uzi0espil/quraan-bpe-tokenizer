# Quraan Tokenizer

I trained a BPE tokenizer with Quraan Uthmani with recitations [Link](https://api.alquran.cloud/v1/quran/quran-uthmani). I eventually saved the BPE in order to be used later for encoding prior to LLM training.

## Installation and Running

1. Clone the repository
2. install `requirements.txt`
3. Check `quraan_tokenizer.ipynb` on how to fit the tokenizer and create dataloader.

Additionally, I saved the final Quraan tokenizer file, you can load it and use it right away for LLM.

## Fitting BPE

Reading Arabic [encodings](https://www.unicode.org/charts/nameslist/n_0600.html).
We will first try to remove some characters that might generate different tokens for the same work:

- Honorifics
    - 0610	 ◌ؐ 	Arabic Sign Sallallahou Alayhe Wassallam
 	 	•	represents sallallahu alayhe wasallam "may God's peace and blessings be upon him"
    - 0611	 ◌ؑ 	Arabic Sign Alayhe Assallam
 	 	•	represents alayhe assalam "upon him be peace"
 	 	→	FD47 ﵇ arabic ligature alayhi as-salaam
    - 0612	 ◌ؒ 	Arabic Sign Rahmatullah Alayhe
 	 	•	represents rahmatullah alayhe "may God have mercy upon him"
 	 	→	FD40 ﵀ arabic ligature rahimahu allaah
    - 0613	 ◌ؓ 	Arabic Sign Radi Allahou Anhu
 	 	•	represents radi allahu 'anhu "may God be pleased with him"
 	 	→	FD41 ﵁ arabic ligature radi allaahu anh
    - 0614	 ◌ؔ 	Arabic Sign Takhallus
 	 	•	sign placed over the name or nom-de-plume of a poet, or in some writings used to mark all proper names
- Extended Arabic mark:
    - 0616	 ◌ؖ 	Arabic Small High Ligature Alef With Lam With Yeh
 	 	※	ARABIC SMALL HIGH LIGATURE ALEF WITH YEH BARREE
 	 	•	early Persian
- Quaranic annotation sign:
    - 06E5	 ‎ۥ‎ 	Arabic Small Waw
 	 	→	08D3 ◌࣓ arabic small low waw
 	 	→	08F3 ◌ࣳ arabic small high waw
    - 0617	 ◌ؗ 	Arabic Small High Zain
    - 0618	 ◌ؘ 	Arabic Small Fatha
    	 	•	should not be confused with 064E ◌َ FATHA
    - 0619	 ◌ؙ 	Arabic Small Damma
 	 	•	should not be confused with 064F ◌ُ DAMMA
    - 061A	 ◌ؚ 	Arabic Small Kasra
 	 	•	should not be confused with 0650 ◌ِ KASRA
    - 06DC	 ◌ۜ 	Arabic Small High Seen
    -  06DE	 ۞ 	Arabic Start Of Rub El Hizb
 	 	•	indicates boundaries of parts of sections
 	 	•	typically depicted as an eight-sided symbol, which may or may not appear starlike
    - 06DF	 ◌۟ 	Arabic Small High Rounded Zero
 	 	•	smaller than the typical circular shape used for 0652 ◌ْ
    - 06E0	 ◌۠ 	Arabic Small High Upright Rectangular Zero
 	 	•	the term "rectangular zero" is a translation of the Arabic name of this sign
    - 06E1	 ◌ۡ 	Arabic Small High Dotless Head Of Khah
 	 	=	Arabic jazm
 	 	•	presentation form of 0652 ◌ْ, using font technology to select the variant is preferred
 	 	•	used in some Qurans to mark absence of a vowel
 	 	→	0652 ◌ْ arabic sukun
    - 06E2	 ◌ۢ 	Arabic Small High Meem Isolated Form
    - 06E3	 ◌ۣ 	Arabic Small Low Seen
    - 06E4	 ◌ۤ 	Arabic Small High Madda
 	 	•	typically used with 06E5 ‎ۥ‎, 06E6 ‎ۦ‎, 06E7 ◌ۧ, and 08F3 ◌ࣳ
    - 06E5	 ‎ۥ‎ 	Arabic Small Waw
 	 	→	08D3 ◌࣓ arabic small low waw
 	 	→	08F3 ◌ࣳ arabic small high waw
    - 06E6	 ‎ۦ‎ 	Arabic Small Yeh
    - 06E7	 ◌ۧ 	Arabic Small High Yeh
    - 06E8	 ◌ۨ 	Arabic Small High Noon
    - 06E9	 ۩ 	Arabic Place Of Sajdah
 	 	•	there is a range of acceptable glyphs for this character
    - 06EA	 ◌۪ 	Arabic Empty Centre Low Stop
    - 06EB	 ◌۫ 	Arabic Empty Centre High Stop
    - 06EC	 ◌۬ 	Arabic Rounded High Stop With Filled Centre
 	 	•	also used in Quranic text in African and other orthographies to represent wasla, ikhtilas, etc.
    - 06ED	 ◌ۭ 	Arabic Small Low Meem
    
Then, we will split following GPT4 pattern but including:

- Arabic words with all recitations.
- 06D6	 ◌ۖ 	Arabic Small High Ligature Sad With Lam With Alef Maksura
- 06D7	 ◌ۗ 	Arabic Small High Ligature Qaf With Lam With Alef Maksura
- 06D8	 ◌ۘ 	Arabic Small High Meem Initial Form
- 06D9	 ◌ۙ 	Arabic Small High Lam Alef
- 06DA	 ◌ۚ 	Arabic Small High Jeem
- 06DB	 ◌ۛ 	Arabic Small High Three Dots

## Encoding

When encoding for training, we need to tell the LLM model to differentiate between Ayat and Surahs, that is why we need to provide a special tokens to split between different ayat and surahs.

- Ayat split token: ۝
- Surah split token: <|endoftext|>
