pwd="""*****"""
aws_access_key="""*****"""
aws_secret_key="""*****"""
api_url="""https://data.gov.il/api/3/action/datastore_search?resource_id=d337959a-020a-4ed3-84f7-fca182292308&limit=1000000&q={}"""
latest_date_query="""SELECT top 1 * from COVID_SYMP_TEST ORDER BY TEST_DATE DESC"""
insert_into_clean_query="""INSERT INTO COVID_SYMP_TEST_CLEAN (TEST_DATE,COUGH,FEVER,SORE_THROAT,SHORTNESS_OF_BREATH,HEAD_ACHE,CORONA_RESULT,AGE_60_AND_ABOVE,GENDER,TEST_INDICATION)
SELECT 
cast(extract(epoch from TEST_DATE) as int),
cast(replace(
    CASE 
        WHEN COUGH = FALSE THEN 0
        WHEN COUGH = TRUE THEN 1
        ELSE 0
    END, 'e', '!'
) as int),
cast(replace(
    CASE 
        WHEN FEVER = FALSE THEN 0
        WHEN FEVER = TRUE THEN 1
        ELSE 0
    END, 'e', '!' 
) as int),
cast(replace(
    CASE 
        WHEN SORE_THROAT = FALSE THEN 0
        WHEN SORE_THROAT = TRUE THEN 1
        ELSE 0
    END, 'e', '!' 
) as int),
cast(replace(
    CASE 
        WHEN SHORTNESS_OF_BREATH = FALSE THEN 0
        WHEN SHORTNESS_OF_BREATH = TRUE THEN 1
        ELSE 0
    END, 'e', '!' 
) as int),
cast(replace(
    CASE 
        WHEN HEAD_ACHE = FALSE THEN 0
        WHEN HEAD_ACHE = TRUE THEN 1
        ELSE 0
    END, 'e', '!' 
) as int),
cast(replace(
    CASE 
        WHEN CORONA_RESULT = 'שלילי' THEN 0
        WHEN CORONA_RESULT = 'חיובי' THEN 1
        ELSE 0
    END, 'e', '!' 
) as int),
cast(replace(
    CASE 
        WHEN AGE_60_AND_ABOVE = 'No' THEN 0
        WHEN AGE_60_AND_ABOVE = 'Yes' THEN 1
        ELSE 0
    END, 'e', '!' 
) as int),
cast(replace(
    CASE 
        WHEN GENDER = 'זכר' THEN 1
        WHEN GENDER = 'נקבה' THEN 0
        ELSE 0
    END, 'e', '!'
) as int),
cast(replace(
    CASE 
        WHEN TEST_INDICATION = 'Other' THEN 0
        WHEN TEST_INDICATION = 'Contact with confirmed' THEN 1
        ELSE 0
    END, 'e', '!' 
) as int)
FROM COVID_SYMP_TEST_STREAM;"""
