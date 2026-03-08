WITH results_until_date AS (
    SELECT *
    FROM `main_2_bronze`.`f1_results`
    WHERE date(date) <= date('2024-04-21')
    ORDER BY date DESC
),
drivers_selected AS (
    SELECT DISTINCT driverid
    FROM results_until_date
    WHERE YEAR >= (
            SELECT MAX(YEAR) - 2
            FROM results_until_date
        )
),
tb_results AS (
    SELECT t1.*
    FROM results_until_date AS t1
        INNER JOIN drivers_selected AS t2 ON t1.driverid = t2.driverid
    ORDER BY YEAR
),
tb_life AS (
    SELECT driverid,
        count(DISTINCT YEAR) AS qtd_seasons,
        count(*) AS qtd_sessions,
        sum(
            CASE
                WHEN (
                    status = 'Finished'
                    OR status LIKE '+%'
                ) THEN 1
                ELSE 0
            END
        ) AS qtde_sessions_finished,
        sum(
            CASE
                WHEN mode = 'Race' THEN 1
                ELSE 0
            END
        ) AS qtd_race,
        sum(
            CASE
                WHEN mode = 'Race'
                AND (
                    status = 'Finished'
                    OR status LIKE '+%'
                ) THEN 1
                ELSE 0
            END
        ) AS qtde_sessions_finished_race,
        sum(
            CASE
                WHEN mode = 'Sprint' THEN 1
                ELSE 0
            END
        ) AS qtd_sprint,
        sum(
            CASE
                WHEN mode = 'Sprint'
                AND (
                    status = 'Finished'
                    OR status LIKE '+%'
                ) THEN 1
                ELSE 0
            END
        ) AS qtde_sessions_finished_sprint,
        sum(
            CASE
                WHEN POSITION = 1 THEN 1
                ELSE 0
            END
        ) AS qtde_1Pos,
        sum(
            CASE
                WHEN POSITION = 1
                AND MODE = 'Race' THEN 1
                ELSE 0
            END
        ) AS qtde_1Pos_race,
        sum(
            CASE
                WHEN POSITION = 1
                AND MODE = 'Sprint' THEN 1
                ELSE 0
            END
        ) AS qtde_1Pos_sprint,
        sum(
            CASE
                WHEN POSITION <= 3 THEN 1
                ELSE 0
            END
        ) AS qtde_podios,
        sum(
            CASE
                WHEN POSITION <= 3
                AND MODE = 'Race' THEN 1
                ELSE 0
            END
        ) AS qtde_podios_race,
        sum(
            CASE
                WHEN POSITION <= 3
                AND MODE = 'Sprint' THEN 1
                ELSE 0
            END
        ) AS qtde_podios_sprint,
        sum(
            CASE
                WHEN POSITION <= 5 THEN 1
                ELSE 0
            END
        ) AS qtde_pos5,
        sum(
            CASE
                WHEN POSITION <= 5
                AND MODE = 'Race' THEN 1
                ELSE 0
            END
        ) AS qtde_pos5_race,
        sum(
            CASE
                WHEN POSITION <= 5
                AND MODE = 'Sprint' THEN 1
                ELSE 0
            END
        ) AS qtde_pos5_sprint,
        sum(
            CASE
                WHEN gridposition <= 5 THEN 1
                ELSE 0
            END
        ) AS qtde_gridpos5,
        sum(
            CASE
                WHEN gridposition <= 5
                AND MODE = 'Race' THEN 1
                ELSE 0
            END
        ) AS qtde_gridpos5_race,
        sum(
            CASE
                WHEN gridposition <= 5
                AND MODE = 'Sprint' THEN 1
                ELSE 0
            END
        ) AS qtde_gridpos5_sprint,
        sum(points) AS qtde_points,
        sum(
            CASE
                WHEN mode = 'Race' THEN points
            END
        ) AS qtde_points_race,
        sum(
            CASE
                WHEN mode = 'Sprint' THEN points
            END
        ) AS qtde_points_sprint,
        avg(gridposition) AS avg_gridposition,
        avg(
            CASE
                WHEN mode = 'Race' THEN gridposition
            END
        ) AS avg_gridposition_race,
        avg(
            CASE
                WHEN mode = 'Sprint' THEN gridposition
            END
        ) AS avg_gridposition_sprint,
        avg(POSITION) AS avg_position,
        avg(
            CASE
                WHEN mode = 'Race' THEN POSITION
            END
        ) AS avg_position_race,
        avg(
            CASE
                WHEN mode = 'Sprint' THEN POSITION
            END
        ) AS avg_position_sprint,
        sum(
            CASE
                WHEN gridposition = 1 THEN 1
                ELSE 0
            END
        ) AS qtde_1_gridposition,
        sum(
            CASE
                WHEN gridposition = 1
                AND mode = 'Race' THEN 1
                ELSE 0
            END
        ) AS qtde_1_gridposition_race,
        sum(
            CASE
                WHEN gridposition = 1
                AND mode = 'Sprint' THEN 1
                ELSE 0
            END
        ) AS qtde_1_gridposition_sprint,
        sum(
            CASE
                WHEN gridposition = 1
                AND POSITION = 1 THEN 1
                ELSE 0
            END
        ) AS qtde_pole_win,
        sum(
            CASE
                WHEN gridposition = 1
                AND POSITION = 1
                AND mode = 'Race' THEN 1
                ELSE 0
            END
        ) AS qtde_pole_win_race,
        sum(
            CASE
                WHEN gridposition = 1
                AND POSITION = 1
                AND mode = 'Sprint' THEN 1
                ELSE 0
            END
        ) AS qtde_pole_win_sprint,
        sum(
            CASE
                WHEN points > 0 THEN 1
                ELSE 0
            END
        ) AS qtd_sessions_points,
        sum(
            CASE
                WHEN mode = 'Race'
                AND points > 0 THEN 1
                ELSE 0
            END
        ) AS qtd_sessions_points_race,
        sum(
            CASE
                WHEN mode = 'Sprint'
                AND points > 0 THEN 1
                ELSE 0
            END
        ) AS qtd_sessions_points_sprint,
        sum(
            CASE
                WHEN POSITION < gridposition THEN 1
                ELSE 0
            END
        ) AS qtd_sessions_with_overtake,
        sum(
            CASE
                WHEN mode = 'Race'
                AND POSITION < gridposition THEN 1
                ELSE 0
            END
        ) AS qtd_sessions_with_overtake_race,
        sum(
            CASE
                WHEN mode = 'Sprint'
                AND POSITION < gridposition THEN 1
                ELSE 0
            END
        ) AS qtd_sessions_with_overtake_sprint,
        avg(gridposition - POSITION) AS avg_overtake,
        avg(
            CASE
                WHEN mode = 'Race' THEN gridposition - POSITION
            END
        ) AS avg_overtake_race,
        avg(
            CASE
                WHEN mode = 'Sprint' THEN gridposition - POSITION
            END
        ) AS avg_overtake_sprint,
        FROM tb_results
    GROUP BY driverid
)
SELECT *
FROM tb_life
ORDER BY driverid