# %%
import dotenv
import os
import nekt

nekt.data_access_token = "PmLtj8DjJQm4SDNeHrFLUeWeiVt99vxJ9EIOtmDlgBpi4hzpdfCinCLFyjSu3VM6aVU5yp4NlLAYV9s3o1bgKWY4XAjhLazzT8NtiIfkYaniAnxGqXqGDtK8PqQgtG1oZLPf8fOoYS9UpiSFekv6g3ByujIrxhx6scPzyVGl4ubnv34MI2WLW5iDmEf2BgtTnbbdH5qkOK5J9lxqry0xxINh5EUT4KUZTHchDUM36rkHvvnOhw3gE5DzjgATFC4V"
spark = nekt.get_spark_session()

(nekt.load_table(layer_name="Silver", table_name="fs_f1_driver_all")
     .createOrReplaceTempView("fs_f1_driver_all"))


(nekt.load_table(layer_name="Silver", table_name="f1_champions")
     .createOrReplaceTempView("f1_champions"))

query = """

WITH tb_abt AS (
    
    SELECT t1.*,
        coalesce(t2.rankdriver,0 ) AS flChampion

    FROM fs_f1_driver_all AS t1

    LEFT JOIN f1_champions AS t2
    ON t1.driverid = t2.driverid
    AND year(t1.dtref) = t2.year

    WHERE t1.dtref >= date('2000-01-01')
    AND t1.dtref < date('2026-01-01')

    order by dtref desc, driverid
)

SELECT * FROM tb_abt

"""

df = spark.sql(query).toPandas()
df.to_csv("../data/abt_f1_drivers_champion.csv", index=False, sep=';')
