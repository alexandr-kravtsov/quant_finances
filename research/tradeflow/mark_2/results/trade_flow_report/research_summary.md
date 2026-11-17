# JERICHON — Trade Flow: compact research export

Сформировано: 2026-11-06 21:58:46

Квантили parquet рассчитаны по детерминированной равномерной выборке до 100 000 значений на колонку.
Count/null/min/max/mean/std рассчитаны по полному файлу. CSV-результаты читаются целиком.

## SMLT

- `SMLT_trade_flow_features.parquet` — 1,516,098 строк, 41 колонок, 138.9 MB
- `SMLT_trade_price_targets.parquet` — 1,516,098 строк, 13 колонок, 12.8 MB
- `SMLT_event_uniqueness.parquet` — 1,516,098 строк, 10 колонок, 33.1 MB
- `SMLT_trades_normalized.parquet` — 1,516,098 строк, 10 колонок, 24.3 MB

### Event overlap / uniqueness

 mean_uniqueness  median_uniqueness  max_concurrency
        0.500008           0.500000                2
        0.250012           0.250000                4
        0.166680           0.166667                6
        0.090923           0.090909               11
        0.038477           0.038462               26
        0.019623           0.019608               51
        0.186778           0.158631             1121
        0.100875           0.087940             1122
        0.068380           0.061352             1659
        0.041473           0.038555             1794

### Bucket results (50 rows)

           signal  bucket  ret_next_1_trades_bps_mean  ret_next_5_trades_bps_mean  ret_next_10_trades_bps_mean  ret_after_1_distinct_price_changes_bps_mean  ret_after_3_distinct_price_changes_bps_mean  ret_after_5_distinct_price_changes_bps_mean
  tfi_notional_10       0                    0.151240                    0.188030                     0.117896                                     0.692764                                     0.074973                                    -0.049932
  tfi_notional_10       4                   -0.171986                   -0.141710                    -0.039692                                    -0.227935                                     0.174798                                     0.522204
 tfi_notional_100       0                   -0.007955                   -0.079551                    -0.150957                                     0.419229                                    -0.205649                                    -0.307943
 tfi_notional_100       4                    0.012923                    0.102719                     0.206644                                     0.013397                                     0.395001                                     0.641327
tfi_notional_1000       0                   -0.006413                   -0.040169                    -0.081500                                     0.122638                                    -0.220033                                    -0.237195
tfi_notional_1000       4                    0.007686                    0.041841                     0.081464                                     0.119652                                     0.215702                                     0.413363
  tfi_notional_25       0                    0.016175                   -0.080224                    -0.199810                                     0.369964                                    -0.357257                                    -0.489063
  tfi_notional_25       4                   -0.015381                    0.110317                     0.270745                                     0.016829                                     0.506359                                     0.888970
  tfi_notional_50       0                   -0.005529                   -0.106651                    -0.221048                                     0.344722                                    -0.375511                                    -0.475616
  tfi_notional_50       4                    0.007797                    0.133936                     0.275761                                     0.046454                                     0.523572                                     0.844889
    tfi_volume_10       0                    0.151267                    0.188049                     0.117881                                     0.692819                                     0.074995                                    -0.049855
    tfi_volume_10       4                   -0.173063                   -0.156874                    -0.071725                                    -0.246919                                     0.116445                                     0.465816
   tfi_volume_100       0                   -0.007904                   -0.079830                    -0.151272                                     0.418697                                    -0.205721                                    -0.308123
   tfi_volume_100       4                    0.013150                    0.102917                     0.206916                                     0.014019                                     0.395118                                     0.641233
  tfi_volume_1000       0                   -0.006324                   -0.040078                    -0.081453                                     0.122665                                    -0.220026                                    -0.237928
  tfi_volume_1000       4                    0.008059                    0.041890                     0.082321                                     0.122074                                     0.217287                                     0.415010
    tfi_volume_25       0                    0.016128                   -0.080173                    -0.199811                                     0.370036                                    -0.357052                                    -0.488870
    tfi_volume_25       4                   -0.015354                    0.110332                     0.270652                                     0.016706                                     0.506201                                     0.888602
    tfi_volume_50       0                   -0.005569                   -0.106547                    -0.221081                                     0.344320                                    -0.375679                                    -0.475883
    tfi_volume_50       4                    0.007895                    0.133923                     0.275705                                     0.046478                                     0.523647                                     0.845249

### Day-block bootstrap (60 tests)

           signal                                 target  spread_bps    ci_low   ci_high  days
  tfi_notional_25 ret_after_5_distinct_price_changes_bps    1.377812  0.687121  2.163413    24
    tfi_volume_25 ret_after_5_distinct_price_changes_bps    1.377691  0.550154  2.110681    24
    tfi_volume_50 ret_after_5_distinct_price_changes_bps    1.321088  0.655740  1.970255    24
  tfi_notional_50 ret_after_5_distinct_price_changes_bps    1.320945  0.663664  1.918267    24
    tfi_volume_10 ret_after_1_distinct_price_changes_bps   -0.954892 -1.851947 -0.183726    24
 tfi_notional_100 ret_after_5_distinct_price_changes_bps    0.950142  0.474068  1.405386    24
   tfi_volume_100 ret_after_5_distinct_price_changes_bps    0.949809  0.452102  1.445519    24
  tfi_notional_10 ret_after_1_distinct_price_changes_bps   -0.921350 -1.721629 -0.210279    24
  tfi_notional_50 ret_after_3_distinct_price_changes_bps    0.899210  0.424072  1.488647    24
    tfi_volume_50 ret_after_3_distinct_price_changes_bps    0.899100  0.429801  1.443217    24
  tfi_notional_25 ret_after_3_distinct_price_changes_bps    0.863423  0.301826  1.530300    24
    tfi_volume_25 ret_after_3_distinct_price_changes_bps    0.863346  0.321558  1.507789    24
  tfi_volume_1000 ret_after_5_distinct_price_changes_bps    0.652866  0.235985  1.025729    24
tfi_notional_1000 ret_after_5_distinct_price_changes_bps    0.651060  0.206431  1.010752    24
   tfi_volume_100 ret_after_3_distinct_price_changes_bps    0.600801  0.198520  1.118696    24

### Momentum control (180 rows)

       signal                target          momentum_control  momentum_bucket  flow_bucket      n  mean_future_bps  median_future_bps  up_probability
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            0 214758         0.780078                0.0        0.241826
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            1 186360         1.214273                0.0        0.339885
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            2 104136         1.088302                0.0        0.302268
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            0 175551        -0.082825                0.0        0.166271
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            1 127736        -0.107178                0.0        0.226827
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            2 201967         0.098797                0.0        0.187808
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            0 114945        -1.209339                0.0        0.178659
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            1 191158        -1.061201                0.0        0.224929
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            2 199151        -0.707556                0.0        0.237714
tfi_volume_10 ret_next_5_trades_bps past_return_20_trades_bps                0            0 304446         0.539247                0.0        0.234524

## VTBR

- `VTBR_trade_flow_features.parquet` — 3,335,582 строк, 41 колонок, 315.1 MB
- `VTBR_trade_price_targets.parquet` — 3,335,582 строк, 13 колонок, 39.5 MB
- `VTBR_event_uniqueness.parquet` — 3,335,582 строк, 10 колонок, 67.1 MB
- `VTBR_trades_normalized.parquet` — 3,335,582 строк, 10 колонок, 63.3 MB

### Event overlap / uniqueness

 mean_uniqueness  median_uniqueness  max_concurrency
        0.500006           0.500000                2
        0.250009           0.250000                4
        0.166676           0.166667                6
        0.090919           0.090909               11
        0.038472           0.038462               26
        0.019619           0.019608               51
        0.224956           0.218571             2722
        0.127480           0.118862             2731
        0.087259           0.082206             3103
        0.053743           0.051963             3114

### Bucket results (50 rows)

           signal  bucket  ret_next_1_trades_bps_mean  ret_next_5_trades_bps_mean  ret_next_10_trades_bps_mean  ret_after_1_distinct_price_changes_bps_mean  ret_after_3_distinct_price_changes_bps_mean  ret_after_5_distinct_price_changes_bps_mean
  tfi_notional_10       0                    0.024856                    0.008348                    -0.012952                                     0.145032                                     0.019521                                    -0.105816
  tfi_notional_10       4                   -0.032107                    0.013741                     0.064268                                    -0.192030                                     0.011458                                     0.156226
 tfi_notional_100       0                   -0.008587                   -0.044766                    -0.079320                                     0.023855                                    -0.099792                                    -0.220794
 tfi_notional_100       4                    0.010415                    0.057157                     0.105418                                    -0.131088                                     0.061923                                     0.205908
tfi_notional_1000       0                   -0.001574                   -0.009280                    -0.017043                                     0.023896                                    -0.026916                                    -0.087173
tfi_notional_1000       4                    0.002050                    0.011265                     0.020442                                    -0.140648                                    -0.023494                                     0.050477
  tfi_notional_25       0                   -0.003037                   -0.041502                    -0.082824                                     0.072370                                    -0.072032                                    -0.210999
  tfi_notional_25       4                    0.004891                    0.059202                     0.115300                                    -0.148324                                     0.057323                                     0.213588
  tfi_notional_50       0                   -0.009146                   -0.054550                    -0.100342                                     0.036245                                    -0.108272                                    -0.246601
  tfi_notional_50       4                    0.010504                    0.065495                     0.119554                                    -0.135833                                     0.063118                                     0.220734
    tfi_volume_10       0                    0.024855                    0.008349                    -0.012950                                     0.145032                                     0.019528                                    -0.105806
    tfi_volume_10       4                   -0.032898                    0.008015                     0.053840                                    -0.208407                                    -0.019673                                     0.116236
   tfi_volume_100       0                   -0.008589                   -0.044780                    -0.079326                                     0.023849                                    -0.099802                                    -0.220822
   tfi_volume_100       4                    0.010403                    0.057168                     0.105430                                    -0.131104                                     0.061939                                     0.205871
  tfi_volume_1000       0                   -0.001590                   -0.009287                    -0.017071                                     0.023878                                    -0.026967                                    -0.087250
  tfi_volume_1000       4                    0.002027                    0.011289                     0.020510                                    -0.140595                                    -0.023484                                     0.050402
    tfi_volume_25       0                   -0.003054                   -0.041519                    -0.082851                                     0.072356                                    -0.072041                                    -0.211003
    tfi_volume_25       4                    0.004884                    0.059206                     0.115311                                    -0.148323                                     0.057335                                     0.213596
    tfi_volume_50       0                   -0.009142                   -0.054532                    -0.100341                                     0.036261                                    -0.108254                                    -0.246607
    tfi_volume_50       4                    0.010495                    0.065489                     0.119532                                    -0.135840                                     0.063091                                     0.220705

### Day-block bootstrap (60 tests)

          signal                                 target  spread_bps    ci_low   ci_high  days
   tfi_volume_50 ret_after_5_distinct_price_changes_bps    0.467307  0.342516  0.585283    38
 tfi_notional_50 ret_after_5_distinct_price_changes_bps    0.467272  0.345000  0.583856    38
tfi_notional_100 ret_after_5_distinct_price_changes_bps    0.426793  0.333196  0.531072    38
  tfi_volume_100 ret_after_5_distinct_price_changes_bps    0.426746  0.329643  0.531743    38
   tfi_volume_25 ret_after_5_distinct_price_changes_bps    0.424593  0.286311  0.566390    38
 tfi_notional_25 ret_after_5_distinct_price_changes_bps    0.424573  0.285732  0.561899    38
   tfi_volume_10 ret_after_1_distinct_price_changes_bps   -0.353085 -0.424559 -0.265261    38
 tfi_notional_10 ret_after_1_distinct_price_changes_bps   -0.337059 -0.423759 -0.248335    38
 tfi_notional_10 ret_after_5_distinct_price_changes_bps    0.262073  0.104662  0.418729    38
   tfi_volume_10 ret_after_5_distinct_price_changes_bps    0.234563  0.105425  0.378609    38
 tfi_notional_25 ret_after_1_distinct_price_changes_bps   -0.220697 -0.301706 -0.135866    38
   tfi_volume_25 ret_after_1_distinct_price_changes_bps   -0.220679 -0.300283 -0.141066    38
   tfi_volume_50                 ret_next_10_trades_bps    0.219887  0.183681  0.257651    38
 tfi_notional_50                 ret_next_10_trades_bps    0.219870  0.182429  0.263404    38
   tfi_volume_25                 ret_next_10_trades_bps    0.198155  0.143935  0.249886    38

### Momentum control (180 rows)

       signal                target          momentum_control  momentum_bucket  flow_bucket      n  mean_future_bps  median_future_bps  up_probability
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            0 499973         0.072936                0.0        0.243741
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            1 413952         0.191422                0.0        0.377254
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            2 197758         0.226076                0.0        0.353579
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            0 388404        -0.035758                0.0        0.158052
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            1 293031        -0.045312                0.0        0.247834
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            2 430248         0.019977                0.0        0.228517
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            0 223306        -0.282552                0.0        0.191818
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            1 404700        -0.174829                0.0        0.258320
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            2 483678        -0.004912                0.0        0.327898
tfi_volume_10 ret_next_5_trades_bps past_return_20_trades_bps                0            0 640554         0.038670                0.0        0.223898

## GAZP

- `GAZP_trade_flow_features.parquet` — 2,269,095 строк, 41 колонок, 210.2 MB
- `GAZP_trade_price_targets.parquet` — 2,269,095 строк, 13 колонок, 28.7 MB
- `GAZP_event_uniqueness.parquet` — 2,269,095 строк, 10 колонок, 44.1 MB
- `GAZP_trades_normalized.parquet` — 2,269,095 строк, 10 колонок, 38.7 MB

### Event overlap / uniqueness

 mean_uniqueness  median_uniqueness  max_concurrency
        0.500006           0.500000                2
        0.250008           0.250000                4
        0.166676           0.166667                6
        0.090919           0.090909               11
        0.038472           0.038462               26
        0.019619           0.019608               51
        0.236975           0.228621              646
        0.132449           0.124365             1027
        0.090970           0.086193             1031
        0.055893           0.054219             1143

### Bucket results (50 rows)

           signal  bucket  ret_next_1_trades_bps_mean  ret_next_5_trades_bps_mean  ret_next_10_trades_bps_mean  ret_after_1_distinct_price_changes_bps_mean  ret_after_3_distinct_price_changes_bps_mean  ret_after_5_distinct_price_changes_bps_mean
  tfi_notional_10       0                    0.010852                   -0.044309                    -0.103433                                    -0.001987                                    -0.197036                                    -0.347977
  tfi_notional_10       4                   -0.006319                    0.069027                     0.142600                                     0.071400                                     0.259091                                     0.405082
 tfi_notional_100       0                   -0.009616                   -0.048689                    -0.087126                                    -0.009296                                    -0.163702                                    -0.278182
 tfi_notional_100       4                    0.011440                    0.058730                     0.104236                                     0.051311                                     0.195304                                     0.296287
tfi_notional_1000       0                   -0.000494                   -0.001859                    -0.000954                                     0.024413                                    -0.020627                                    -0.052388
tfi_notional_1000       4                    0.001102                    0.005787                     0.009117                                    -0.009623                                     0.036673                                     0.087016
  tfi_notional_25       0                   -0.011313                   -0.074166                    -0.137231                                    -0.030361                                    -0.227178                                    -0.380011
  tfi_notional_25       4                    0.015074                    0.091529                     0.165822                                     0.085321                                     0.280929                                     0.431850
  tfi_notional_50       0                   -0.013032                   -0.069177                    -0.122831                                    -0.023241                                    -0.204020                                    -0.340911
  tfi_notional_50       4                    0.014923                    0.080289                     0.142346                                     0.068388                                     0.245945                                     0.381719
    tfi_volume_10       0                    0.010852                   -0.044315                    -0.103442                                    -0.001994                                    -0.197041                                    -0.347977
    tfi_volume_10       4                   -0.006321                    0.069035                     0.142602                                     0.071404                                     0.259097                                     0.405082
   tfi_volume_100       0                   -0.009606                   -0.048694                    -0.087158                                    -0.009288                                    -0.163740                                    -0.278207
   tfi_volume_100       4                    0.011445                    0.058778                     0.104282                                     0.051354                                     0.195394                                     0.296362
  tfi_volume_1000       0                   -0.000520                   -0.001947                    -0.001036                                     0.024382                                    -0.020699                                    -0.052464
  tfi_volume_1000       4                    0.001120                    0.005701                     0.009042                                    -0.009592                                     0.036600                                     0.086872
    tfi_volume_25       0                   -0.011335                   -0.074186                    -0.137256                                    -0.030395                                    -0.227243                                    -0.380069
    tfi_volume_25       4                    0.015085                    0.091548                     0.165836                                     0.085331                                     0.280966                                     0.431881
    tfi_volume_50       0                   -0.013052                   -0.069159                    -0.122797                                    -0.023254                                    -0.204013                                    -0.340892
    tfi_volume_50       4                    0.014920                    0.080412                     0.142420                                     0.068411                                     0.246029                                     0.381790

### Day-block bootstrap (60 tests)

          signal                                 target  spread_bps   ci_low  ci_high  days
   tfi_volume_25 ret_after_5_distinct_price_changes_bps    0.811862 0.719018 0.924160    25
 tfi_notional_25 ret_after_5_distinct_price_changes_bps    0.811856 0.714155 0.924406    25
   tfi_volume_10 ret_after_5_distinct_price_changes_bps    0.753043 0.643332 0.866743    25
 tfi_notional_10 ret_after_5_distinct_price_changes_bps    0.753035 0.637291 0.870657    25
   tfi_volume_50 ret_after_5_distinct_price_changes_bps    0.722735 0.640074 0.820849    25
 tfi_notional_50 ret_after_5_distinct_price_changes_bps    0.722670 0.635785 0.815479    25
  tfi_volume_100 ret_after_5_distinct_price_changes_bps    0.574538 0.503876 0.641309    25
tfi_notional_100 ret_after_5_distinct_price_changes_bps    0.574506 0.510376 0.651229    25
   tfi_volume_25 ret_after_3_distinct_price_changes_bps    0.508170 0.427571 0.594415    25
 tfi_notional_25 ret_after_3_distinct_price_changes_bps    0.508107 0.429927 0.599837    25
   tfi_volume_10 ret_after_3_distinct_price_changes_bps    0.456132 0.362944 0.569527    25
 tfi_notional_10 ret_after_3_distinct_price_changes_bps    0.456127 0.354998 0.565128    25
   tfi_volume_50 ret_after_3_distinct_price_changes_bps    0.450063 0.377611 0.525414    25
 tfi_notional_50 ret_after_3_distinct_price_changes_bps    0.449979 0.378492 0.531766    25
  tfi_volume_100 ret_after_3_distinct_price_changes_bps    0.359129 0.308835 0.419458    25

### Momentum control (180 rows)

       signal                target          momentum_control  momentum_bucket  flow_bucket      n  mean_future_bps  median_future_bps  up_probability
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            0 348524         0.001514                0.0        0.250407
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            1 280350         0.094945                0.0        0.354275
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            2 127374         0.239731                0.0        0.413161
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            0 276942        -0.068851                0.0        0.171382
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            1 200329        -0.013759                0.0        0.277149
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            2 278977         0.073714                0.0        0.274772
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            0 130782        -0.263850                0.0        0.227279
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            1 275569        -0.116010                0.0        0.301402
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            2 349898         0.021024                0.0        0.382740
tfi_volume_10 ret_next_5_trades_bps past_return_20_trades_bps                0            0 439311        -0.036357                0.0        0.236029

## ROSN

- `ROSN_trade_flow_features.parquet` — 2,763,301 строк, 41 колонок, 255.6 MB
- `ROSN_trade_price_targets.parquet` — 2,763,301 строк, 13 колонок, 31.0 MB
- `ROSN_event_uniqueness.parquet` — 2,763,301 строк, 10 колонок, 55.7 MB
- `ROSN_trades_normalized.parquet` — 2,763,301 строк, 10 колонок, 48.4 MB

### Event overlap / uniqueness

 mean_uniqueness  median_uniqueness  max_concurrency
        0.500007           0.500000                2
        0.250010           0.250000                4
        0.166678           0.166667                6
        0.090922           0.090909               11
        0.038475           0.038462               26
        0.019621           0.019608               51
        0.219413           0.205556             2006
        0.122210           0.111255             2115
        0.083691           0.077668             2118
        0.051384           0.049276             2700

### Bucket results (50 rows)

           signal  bucket  ret_next_1_trades_bps_mean  ret_next_5_trades_bps_mean  ret_next_10_trades_bps_mean  ret_after_1_distinct_price_changes_bps_mean  ret_after_3_distinct_price_changes_bps_mean  ret_after_5_distinct_price_changes_bps_mean
  tfi_notional_10       0                    0.017636                   -0.049455                    -0.116277                                    -0.061595                                    -0.339621                                    -0.502098
  tfi_notional_10       4                   -0.018329                    0.078717                     0.162312                                     0.013452                                     0.274308                                     0.420189
 tfi_notional_100       0                   -0.011015                   -0.057178                    -0.103936                                    -0.070115                                    -0.294704                                    -0.448688
 tfi_notional_100       4                    0.011937                    0.062824                     0.111292                                    -0.021692                                     0.179136                                     0.309611
tfi_notional_1000       0                   -0.003248                   -0.016248                    -0.030976                                    -0.016702                                    -0.105096                                    -0.173312
tfi_notional_1000       4                    0.001699                    0.009170                     0.015150                                    -0.071932                                     0.003786                                     0.072103
  tfi_notional_25       0                   -0.011861                   -0.082191                    -0.147492                                    -0.096036                                    -0.382161                                    -0.550801
  tfi_notional_25       4                    0.014152                    0.102384                     0.178733                                     0.026943                                     0.283911                                     0.436498
  tfi_notional_50       0                   -0.014638                   -0.080003                    -0.143024                                    -0.090856                                    -0.360241                                    -0.532698
  tfi_notional_50       4                    0.014223                    0.080408                     0.140267                                     0.003527                                     0.234550                                     0.374233
    tfi_volume_10       0                    0.017644                   -0.049448                    -0.116247                                    -0.061572                                    -0.339591                                    -0.502051
    tfi_volume_10       4                   -0.018326                    0.078717                     0.162312                                     0.013454                                     0.274306                                     0.420194
   tfi_volume_100       0                   -0.011032                   -0.057180                    -0.103953                                    -0.070113                                    -0.294672                                    -0.448625
   tfi_volume_100       4                    0.011914                    0.062791                     0.111300                                    -0.021729                                     0.179127                                     0.309615
  tfi_volume_1000       0                   -0.003223                   -0.016166                    -0.030780                                    -0.016592                                    -0.104931                                    -0.172947
  tfi_volume_1000       4                    0.001702                    0.009261                     0.015215                                    -0.071823                                     0.003830                                     0.071988
    tfi_volume_25       0                   -0.011846                   -0.082162                    -0.147479                                    -0.096006                                    -0.382165                                    -0.550798
    tfi_volume_25       4                    0.014146                    0.102387                     0.178738                                     0.026937                                     0.283904                                     0.436484
    tfi_volume_50       0                   -0.014661                   -0.080044                    -0.143069                                    -0.090901                                    -0.360261                                    -0.532727
    tfi_volume_50       4                    0.014216                    0.080394                     0.140201                                     0.003508                                     0.234483                                     0.374190

### Day-block bootstrap (60 tests)

          signal                                 target  spread_bps   ci_low  ci_high  days
   tfi_volume_25 ret_after_5_distinct_price_changes_bps    0.987124 0.752688 1.208117    38
 tfi_notional_25 ret_after_5_distinct_price_changes_bps    0.987113 0.728039 1.192255    38
   tfi_volume_10 ret_after_5_distinct_price_changes_bps    0.922602 0.682636 1.170765    38
 tfi_notional_10 ret_after_5_distinct_price_changes_bps    0.922554 0.700707 1.165057    38
   tfi_volume_50 ret_after_5_distinct_price_changes_bps    0.906319 0.727092 1.077362    38
 tfi_notional_50 ret_after_5_distinct_price_changes_bps    0.906277 0.691207 1.079950    38
tfi_notional_100 ret_after_5_distinct_price_changes_bps    0.758533 0.577353 0.925740    38
  tfi_volume_100 ret_after_5_distinct_price_changes_bps    0.758381 0.559425 0.924143    38
 tfi_notional_25 ret_after_3_distinct_price_changes_bps    0.666129 0.478083 0.847908    38
   tfi_volume_25 ret_after_3_distinct_price_changes_bps    0.666107 0.496847 0.846009    38
   tfi_volume_10 ret_after_3_distinct_price_changes_bps    0.614326 0.431486 0.794024    38
 tfi_notional_10 ret_after_3_distinct_price_changes_bps    0.614318 0.398798 0.836178    38
   tfi_volume_50 ret_after_3_distinct_price_changes_bps    0.594661 0.432385 0.723185    38
 tfi_notional_50 ret_after_3_distinct_price_changes_bps    0.594569 0.430236 0.736158    38
  tfi_volume_100 ret_after_3_distinct_price_changes_bps    0.473957 0.341448 0.590517    38

### Momentum control (180 rows)

       signal                target          momentum_control  momentum_bucket  flow_bucket      n  mean_future_bps  median_future_bps  up_probability
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            0 403664         0.055790                0.0        0.233947
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            1 350487         0.194057                0.0        0.352775
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            2 166772         0.309910                0.0        0.373768
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            0 339395        -0.105400                0.0        0.151434
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            1 228793        -0.043639                0.0        0.267259
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            2 352735         0.101153                0.0        0.250262
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            0 177864        -0.364635                0.0        0.184883
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            1 341643        -0.202741                0.0        0.269225
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            2 401416        -0.004165                0.0        0.321502
tfi_volume_10 ret_next_5_trades_bps past_return_20_trades_bps                0            0 534721        -0.016833                0.0        0.221263

## SBER

- `SBER_trade_flow_features.parquet` — 4,539,739 строк, 41 колонок, 415.9 MB
- `SBER_trade_price_targets.parquet` — 4,539,739 строк, 13 колонок, 58.7 MB
- `SBER_event_uniqueness.parquet` — 4,539,739 строк, 10 колонок, 88.6 MB
- `SBER_trades_normalized.parquet` — 4,539,739 строк, 10 колонок, 79.7 MB

### Event overlap / uniqueness

 mean_uniqueness  median_uniqueness  max_concurrency
        0.500004           0.500000                2
        0.250006           0.250000                4
        0.166674           0.166667                6
        0.090917           0.090909               11
        0.038470           0.038462               26
        0.019616           0.019608               51
        0.225287           0.218571             1025
        0.131539           0.125926             1320
        0.089926           0.086218             1848
        0.055663           0.054632             2064

### Bucket results (50 rows)

           signal  bucket  ret_next_1_trades_bps_mean  ret_next_5_trades_bps_mean  ret_next_10_trades_bps_mean  ret_after_1_distinct_price_changes_bps_mean  ret_after_3_distinct_price_changes_bps_mean  ret_after_5_distinct_price_changes_bps_mean
  tfi_notional_10       0                    0.001869                   -0.033519                    -0.069390                                     0.091620                                    -0.040905                                    -0.132766
  tfi_notional_10       4                   -0.005934                    0.036984                     0.077064                                    -0.074631                                     0.059873                                     0.153091
 tfi_notional_100       0                   -0.006651                   -0.034621                    -0.064537                                     0.044903                                    -0.050993                                    -0.131140
 tfi_notional_100       4                    0.006798                    0.036336                     0.067024                                    -0.040984                                     0.054130                                     0.134470
tfi_notional_1000       0                   -0.002051                   -0.010072                    -0.018697                                     0.033585                                    -0.016430                                    -0.056508
tfi_notional_1000       4                    0.002440                    0.012499                     0.024046                                    -0.043179                                     0.013162                                     0.058464
  tfi_notional_25       0                   -0.006164                   -0.041945                    -0.077919                                     0.061857                                    -0.059777                                    -0.149263
  tfi_notional_25       4                    0.006213                    0.047173                     0.085344                                    -0.050988                                     0.071109                                     0.161132
  tfi_notional_50       0                   -0.007069                   -0.039363                    -0.072760                                     0.052856                                    -0.056268                                    -0.143006
  tfi_notional_50       4                    0.007080                    0.041577                     0.075470                                    -0.048075                                     0.061421                                     0.146845
    tfi_volume_10       0                    0.001865                   -0.033521                    -0.069388                                     0.091620                                    -0.040899                                    -0.132765
    tfi_volume_10       4                   -0.005935                    0.036981                     0.077062                                    -0.074632                                     0.059873                                     0.153089
   tfi_volume_100       0                   -0.006647                   -0.034644                    -0.064541                                     0.044914                                    -0.050969                                    -0.131134
   tfi_volume_100       4                    0.006801                    0.036341                     0.067035                                    -0.040975                                     0.054138                                     0.134489
  tfi_volume_1000       0                   -0.002044                   -0.010096                    -0.018748                                     0.033571                                    -0.016456                                    -0.056483
  tfi_volume_1000       4                    0.002449                    0.012501                     0.024075                                    -0.043171                                     0.013178                                     0.058460
    tfi_volume_25       0                   -0.006162                   -0.041941                    -0.077914                                     0.061859                                    -0.059784                                    -0.149272
    tfi_volume_25       4                    0.006212                    0.047175                     0.085344                                    -0.050990                                     0.071111                                     0.161141
    tfi_volume_50       0                   -0.007070                   -0.039361                    -0.072765                                     0.052852                                    -0.056273                                    -0.143008
    tfi_volume_50       4                    0.007087                    0.041579                     0.075483                                    -0.048065                                     0.061426                                     0.146854

### Day-block bootstrap (60 tests)

          signal                                 target  spread_bps    ci_low   ci_high  days
   tfi_volume_25 ret_after_5_distinct_price_changes_bps    0.310425  0.264386  0.363866    38
 tfi_notional_25 ret_after_5_distinct_price_changes_bps    0.310411  0.264585  0.358365    38
   tfi_volume_50 ret_after_5_distinct_price_changes_bps    0.289860  0.253482  0.335432    38
 tfi_notional_50 ret_after_5_distinct_price_changes_bps    0.289855  0.248780  0.334582    38
   tfi_volume_10 ret_after_5_distinct_price_changes_bps    0.285876  0.231832  0.347130    38
 tfi_notional_10 ret_after_5_distinct_price_changes_bps    0.285874  0.228439  0.346379    38
tfi_notional_100 ret_after_5_distinct_price_changes_bps    0.265603  0.231981  0.299675    38
  tfi_volume_100 ret_after_5_distinct_price_changes_bps    0.265596  0.231516  0.303820    38
 tfi_notional_10 ret_after_1_distinct_price_changes_bps   -0.166255 -0.210083 -0.126389    38
   tfi_volume_10 ret_after_1_distinct_price_changes_bps   -0.166249 -0.210119 -0.119129    38
   tfi_volume_25                 ret_next_10_trades_bps    0.163274  0.145333  0.182034    38
 tfi_notional_25                 ret_next_10_trades_bps    0.163269  0.145344  0.185406    38
   tfi_volume_50                 ret_next_10_trades_bps    0.148243  0.133138  0.163261    38
 tfi_notional_50                 ret_next_10_trades_bps    0.148237  0.131967  0.163852    38
 tfi_notional_10                 ret_next_10_trades_bps    0.146459  0.126276  0.174054    38

### Momentum control (180 rows)

       signal                target          momentum_control  momentum_bucket  flow_bucket      n  mean_future_bps  median_future_bps  up_probability
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            0 652794        -0.028067                0.0        0.241223
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            1 556835         0.071733                0.0        0.384072
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                0            2 303440         0.082855                0.0        0.329498
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            0 530694        -0.012425                0.0        0.136941
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            1 396758        -0.016871                0.0        0.212298
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                1            2 585617         0.006496                0.0        0.151901
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            0 329581        -0.096924                0.0        0.176014
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            1 559476        -0.057259                0.0        0.237299
tfi_volume_10 ret_next_5_trades_bps  past_return_5_trades_bps                2            2 624012         0.046189                0.0        0.341014
tfi_volume_10 ret_next_5_trades_bps past_return_20_trades_bps                0            0 801916        -0.033688                0.0        0.232829

## Interpretation notes

- `bucket_high_minus_low`: среднее значение target в верхнем bucket минус нижний; это описательный эффект, не тест значимости.
- `day_block_bootstrap`: разница верхнего/нижнего квинтилей и доверительный интервал bootstrap по торговым дням.
- CI, не пересекающий 0, — предварительный статистический сигнал, но при малом числе дней он нестабилен.
- `distributions.csv`: моменты и экстремумы рассчитаны по всем строкам; квантили — по выборке до 100 тыс. наблюдений на колонку.
- `trades_normalized.parquet` учитывается по размеру/числу строк, но не профилируется, чтобы не раздувать время и отчёт.
- Эти данные сами по себе не доказывают HFT-альфу. Нужно подтвердить смысл BUYSELL, порядок событий, bid/ask bounce и реалистичность исполнения/издержек.

## Состав экспорта

1. `research_summary.md` — обзор по тикерам и наиболее заметные результаты.
2. `key_findings.csv` — все bucket spreads и day-block bootstrap оценки.
3. `distributions.csv` — статистики/квантили признаков, target и uniqueness.
