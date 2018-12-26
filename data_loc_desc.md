Structure
---------
The train data is seperated by class [0 or 1] to make it easier for training GANs


```
campaign/csv_reservoir
	| train
	    |- [data_class0 & data_class 1].csv/*.csv.gz
	 | validation
	    |- data.csv/*.csv.gz
	 | test
	    |- data.csv/*.csv.gz
```


Features to be Used
--------------------
```
cat_columns = ["exchange_id", "user_frequency", "site_id", "deal_id",
                 "channel_type", "size", "week_part",
                 "day_of_week", "dma_id", "isp_id", "fold_position",
                 "browser_language_id", "country_id", "conn_speed", "os_id",
                 "day_part",
                 "region_id", "browser_id", "hashed_app_id",
                 "interstitial", "device_id", "creative_id",
                 "browser", "browser_version", "os", "os_version",
                 "device_model",
                 "device_manufacturer", "device_type", "exchange_id_cs_vcr",
                 "exchange_id_cs_vrate",
                 "exchange_id_cs_ctr", "exchange_id_cs_category_id",
                 "exchange_id_cs_site_id",
                 "category_id", "cookieless", "cross_device"]

  numeric_columns = ["id_vintage", 
                     "exchange_viewability_rate", "exchange_ctr",
                     "exchange_vcr", *_bpr, *_bpf, *_pixel]
                     
                     
  target = "conversion_target"
  
  Ignore "column_weights"
  
```
