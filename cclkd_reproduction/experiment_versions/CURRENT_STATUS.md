# Current CCLKD Training Status

Pulled from 4090 server at `2026-06-08 21:34:17`.

## GPU

|gpu|mem_used_MiB|mem_free_MiB|util_%|
|---|---|---|---|
|0|12450|11632|98|
|1|12586|11496|98|

## Processes

|run|pid|alive|etime|cpu|mem|
|---|---|---|---|---|---|
|lld|104239|True|12:58:45|103|1.1|
|lld_fld|104551|True|12:58:03|103|1.1|
|ccl_only_cclanchorfix|114333|True|06:25:20|104|0.8|
|lld_fld_rld_rldpatmfix|115163|True|06:14:59|106|0.8|
|atkd_rldpatmfix|115169|True|06:14:59|106|0.8|
|full_rldpatmfix|115175|True|06:14:59|106|0.8|

## Latest Valid/Superseding Curves

|run|status|latest_epoch|latest_AP50|latest_AP|latest_KD|mtime|note|
|---|---|---|---|---|---|---|---|
|lld|running|304|0.751658|0.493256|0.471530|2026-06-08 21:31:49|valid: LLD-only curve; no CCL/RLD/PATM dependency|
|lld_fld|running|287|0.743759|0.486046|0.642010|2026-06-08 21:32:45|valid: LLD+FLD fixed-temperature curve; no CCL/RLD/PATM dependency|
|ccl_only_cclanchorfix|running|150|0.644947|0.397256|0.053320|2026-06-08 21:31:41|valid current CCL-only curve|
|lld_fld_rld_rldpatmfix|running|146|0.646752|0.395538|1.245440|2026-06-08 21:31:48|valid current LLD+FLD+RLD curve|
|atkd_rldpatmfix|running|146|0.646546|0.390955|3.955070|2026-06-08 21:33:17|valid current ATKD curve|
|full_rldpatmfix|running|150|0.643886|0.392374|4.173890|2026-06-08 21:33:23|valid current full CCLKD curve|

Status is a snapshot; runs may continue past these epochs after this file is committed.
