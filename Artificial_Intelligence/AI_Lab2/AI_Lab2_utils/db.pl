% Prolog Cell: Part 1b – Similarity Rules with 1–5 Scales (Full Version)
% ============================================================

:- dynamic director/2, genre/2, actor/2, keyword/2, language/2,
           spoken_language/2, studio/2, country/2, production_country/2,
           release_year/2, duration/2, budget/2, gross/2, popularity/2,
           imdb_rating/2, num_votes/2, similarity_target/1.

% Memoise heavy set-construction rules
:- table common_genres/3, common_keywords/3, common_actors/3,
         common_spoken_languages/3, common_production_countries/3.

% ------------------------------------------------------------------
% 0. Explicit “exact / fairly / relative” helpers
% ------------------------------------------------------------------

% —— Theme (genres) ———————————————————————————
exact_common_theme(M1,M2)       :- genre_score(M1,M2,S), S>=4.
fairly_common_theme(M1,M2)      :- genre_score(M1,M2,S), S>=2, S=<3.
relative_common_theme(M1,M2)    :- genre_score(M1,M2,S), S=:=1.

% —— Plot keywords ————————————————————————
exact_same_plot(M1,M2)          :- keyword_score(M1,M2,S), S>=4.
fairly_similar_plot(M1,M2)      :- keyword_score(M1,M2,S), S>=2, S=<3.
relative_similar_plot(M1,M2)    :- keyword_score(M1,M2,S), S=:=1.

% —— Cast (main actors) ——————————————————————
same_main_actors(M1,M2)         :- actor_score(M1,M2,S), S>=3.
fairly_similar_actors(M1,M2)    :- actor_score(M1,M2,S), S=:=2.
relatively_similar_actors(M1,M2):- actor_score(M1,M2,S), S=:=1.

% ------------------------------------------------------------------
% 1. Extract shared features (sets of common values)
% ------------------------------------------------------------------

common_genres(M1,M2,Common)              :- findall(G,(genre(M1,G),genre(M2,G)),L), sort(L,Common).
common_keywords(M1,M2,Common)            :- findall(K,(keyword(M1,K),keyword(M2,K)),L), sort(L,Common).
common_actors(M1,M2,Common)              :- findall(A,(actor(M1,A),actor(M2,A)),L), sort(L,Common).
common_spoken_languages(M1,M2,Common)    :- findall(L,(spoken_language(M1,L),spoken_language(M2,L)),L0), sort(L0,Common).
common_production_countries(M1,M2,Common):- findall(C,(production_country(M1,C),production_country(M2,C)),L), sort(L,Common).

% ------------------------------------------------------------------
% 2. Count-based similarity scores (clamped to 5)
% ------------------------------------------------------------------

genre_score(M1,M2,S)              :- common_genres(M1,M2,C),              length(C,N), (N<5->S=N;S=5).
keyword_score(M1,M2,S)            :- common_keywords(M1,M2,C),            length(C,N), (N<5->S=N;S=5).
actor_score(M1,M2,S)              :- common_actors(M1,M2,C),              length(C,N), (N<5->S=N;S=5).
spoken_language_score(M1,M2,S)    :- common_spoken_languages(M1,M2,C),    length(C,N), (N<5->S=N;S=5).
production_country_score(M1,M2,S) :- common_production_countries(M1,M2,C),length(C,N), (N<5->S=N;S=5).

% ------------------------------------------------------------------
% 3. Numeric difference-based similarity scores
% ------------------------------------------------------------------

year_score(M1,M2,S)    :- release_year(M1,Y1), release_year(M2,Y2),
                          Diff is abs(Y1-Y2), DecGap is Diff//10,
                          Tmp is 5-DecGap, (Tmp>0->S=Tmp;S=0).

runtime_score(M1,M2,S) :- duration(M1,D1), duration(M2,D2),
                          Diff is abs(D1-D2)//10, Tmp is 5-Diff,
                          (Tmp>0->S=Tmp;S=0).

budget_score(M1,M2,S)  :- budget(M1,B1), budget(M2,B2),
                          (B1=<0;B2=<0->S=0;
                           (B1<B2->Min=B1,Max=B2;Min=B2,Max=B1),
                           Ratio is Min/Max, Tmp is floor(Ratio*5), S=Tmp).

popularity_score(M1,M2,S):- popularity(M1,P1), popularity(M2,P2),
                            (P1=<0;P2=<0->S=0;
                             (P1<P2->Min=P1,Max=P2;Min=P2,Max=P1),
                             Ratio is Min/Max, Tmp is floor(Ratio*5), S=Tmp).

gross_score(M1,M2,S)   :- gross(M1,G1), gross(M2,G2),
                          (G1=<0;G2=<0->S=0;
                           (G1<G2->Min=G1,Max=G2;Min=G2,Max=G1),
                           Ratio is Min/Max, Tmp is floor(Ratio*5), S=Tmp).

rating_score(M1,M2,S)  :- imdb_rating(M1,R1), imdb_rating(M2,R2),
                          Diff is abs(R1-R2), Step is round(Diff/2),
                          Tmp is 5-Step, (Tmp>0->S=Tmp;S=0).

votes_score(M1,M2,S)   :- num_votes(M1,V1), num_votes(M2,V2),
                          (V1=<0;V2=<0->S=0;
                           (V1<V2->Min=V1,Max=V2;Min=V2,Max=V1),
                           Ratio is Min/Max, Tmp is floor(Ratio*5), S=Tmp).

% ------------------------------------------------------------------
% 4. Binary match indicators (true/false)
% ------------------------------------------------------------------

% True if movies share same director (but are not identical)
same_director(M1,M2) :- director(M1,D), director(M2,D), M1\=M2.

% True if movies share same original language
same_language(M1,M2) :- language(M1,L), language(M2,L), M1\=M2.

% —— colour predicate removed (no colour field in dataset) ——
% % True if movies share same colour format                 (REMOVED)
% % same_color(M1,M2) :- color(M1,C), color(M2,C), M1\=M2.

% True if movies share same production studio
same_studio(M1,M2) :- studio(M1,S), studio(M2,S), M1\=M2.

% True if movies share same primary country
same_country(M1,M2) :- country(M1,C), country(M2,C), M1\=M2.

% True if movies released in the same decade
same_decade(M1,M2) :-
    release_year(M1,Y1), release_year(M2,Y2),
    Dec1 is (Y1//10)*10, Dec2 is (Y2//10)*10,
    Dec1 =:= Dec2, Dec1 > 0, M1 \= M2.

% ------------------------------------------------------------------
% 5. Scoring helpers for normalised math
% ------------------------------------------------------------------

clamp_to_5(Raw,S) :-
    (Raw =< 0 -> S = 0
    ; Raw >= 5 -> S = 5
    ; S = Raw), !.            % cut – deterministic

ratio_score(V1,V2,S) :-
    (V1 =< 0 ; V2 =< 0 -> S = 0
    ; (V1 < V2 -> Min = V1, Max = V2 ; Min = V2, Max = V1),
      R is round((Min/Max)*5),
      clamp_to_5(R,S)), !.

difference_score(V1,V2,Step,Scale,S) :-
    D   is abs(V1 - V2) // Step,
    Raw is Scale - D,
    clamp_to_5(Raw,S), !.

% ------------------------------------------------------------------
% 6. Basic composite similarities (0–100)
% ------------------------------------------------------------------

basic_sim_1(M1,M2,S) :- genre_score(M1,M2,G),               actor_score(M1,M2,A),             S is G*10 + A*10.
basic_sim_2(M1,M2,S) :- genre_score(M1,M2,G), keyword_score(M1,M2,K), actor_score(M1,M2,A), S is G*8  + K*6 + A*6.
basic_sim_3(M1,M2,S) :- year_score(M1,M2,Y),  runtime_score(M1,M2,R), rating_score(M1,M2,I),
                                                             budget_score(M1,M2,B),           S is Y*8  + R*8 + I*4 + B*4.

% ------------------------------------------------------------------
% 7. Advanced composite similarities (0–100+)
% ------------------------------------------------------------------

adv_sim_1(M1,M2,S) :-
    genre_score(M1,M2,G), keyword_score(M1,M2,K), actor_score(M1,M2,A),
    rating_score(M1,M2,R), year_score(M1,M2,Y),
    S is G*3 + K*2 + A*3 + R*2 + Y*1.

adv_sim_2(M1,M2,S) :-
    adv_sim_1(M1,M2,B),
    spoken_language_score(M1,M2,L), popularity_score(M1,M2,P),
    S is B + L*5 + P*5.

adv_sim_3(M1,M2,S) :-
    genre_score(M1,M2,G), keyword_score(M1,M2,K), actor_score(M1,M2,A),
    spoken_language_score(M1,M2,L), production_country_score(M1,M2,C),
    rating_score(M1,M2,R), popularity_score(M1,M2,P),
    year_score(M1,M2,Y), runtime_score(M1,M2,Run),
    budget_score(M1,M2,Bu), gross_score(M1,M2,Gr), votes_score(M1,M2,V),
    S is G*3 + K*2 + A*3 + L*2 + C*1 +
         R*2 + P*2 + Y*1 + Run*1 + Bu*1 + Gr*1 + V*1.

% ------------------------------------------------------------------
% 8. Wrapper & tier helper
% ------------------------------------------------------------------

% ★ choose active model here:
similarity_target(adv_sim_1).

similar(M1,M2,S) :-
    similarity_target(Pred),
    M1 @< M2,
    Goal =.. [Pred,M1,M2,S],
    call(Goal).

tier_bounds(5,80,101).  tier_bounds(4,60,80).
tier_bounds(3,40,60).   tier_bounds(2,20,40).
tier_bounds(1,1,20).

find_sim_tier(Level,M1,M2) :-
    tier_bounds(Level,Min,Max),
    similar(M1,M2,S),
    S >= Min, S < Max.

% ------------------------------------------------------------------
% 9. Optional indexing for performance
% ------------------------------------------------------------------

:- index(similar/3,f).
:- index(find_sim_tier/3,f).

% ——— End of file ———
