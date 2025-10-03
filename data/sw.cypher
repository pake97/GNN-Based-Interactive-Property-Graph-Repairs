// :param url=>"https://vbatushkov.bitbucket.io/swapi.json"
:param url=>"file:///swapi.json";

CREATE CONSTRAINT film_url FOR (f:Film) REQUIRE f.url IS UNIQUE;
CREATE CONSTRAINT c_url FOR (c:Character) REQUIRE c.url IS UNIQUE;
CREATE CONSTRAINT p_url FOR (p:Planet) REQUIRE p.url IS UNIQUE;
CREATE CONSTRAINT s_url FOR (s:Starship) REQUIRE s.url IS UNIQUE;
CREATE CONSTRAINT v_url FOR (v:Vehicle) REQUIRE v.url IS UNIQUE;
CREATE CONSTRAINT sp_url FOR (sp:Species) REQUIRE sp.url IS UNIQUE;

// todo remove arrays / relationships from data that's set
CALL apoc.load.json($url) YIELD value
FOREACH (film in value.films | MERGE (f:Film { url: film.url }) ON CREATE SET f += film {.*, characters:null, planets:null, species:null,starships:null, vehicles:null})
FOREACH (character in value.people | MERGE (c:Character { url: character.url }) ON CREATE SET c += character {.*, species:null, vehicles:null, starships:null, films:null, homeworld:null} )
FOREACH (planet in value.planets | MERGE (p:Planet { url: planet.url }) ON CREATE SET p += planet {.*, residents:null, films:null} )
FOREACH (spec in value.species | MERGE (s:Species { url: spec.url }) ON CREATE SET s += spec {.*, films:null, people:null, homeworld:null})
FOREACH (vehicle in value.vehicles | MERGE (v:Vehicle { url: vehicle.url }) ON CREATE SET v += vehicle {.*, pilots:null, films:null} )
FOREACH (starship in value.starships | MERGE (s:Starship { url: starship.url }) ON CREATE SET s += starship {.*, pilots:null, films:null} )
;


CALL apoc.load.json($url) YIELD value
UNWIND value.films as film
UNWIND film.characters as character_url
MATCH (f:Film { url: film.url })
MATCH (c:Character { url: character_url })
MERGE (c)-[:APPEARED_IN]->(f);


CALL apoc.load.json($url) YIELD value
UNWIND value.films as film
UNWIND film.planets as planet_url
MATCH (f:Film { url: film.url })
MATCH (p:Planet { url: planet_url })
MERGE (p)-[:APPEARED_IN]->(f);

CALL apoc.load.json($url) YIELD value
UNWIND value.films as film
UNWIND film.species as species_url
MATCH (f:Film { url: film.url })
MATCH (spec:Species { url: species_url })
MERGE (spec)-[:APPEARED_IN]->(f);

CALL apoc.load.json($url) YIELD value
UNWIND value.films as film
UNWIND film.starships as starship_url
MATCH (f:Film { url: film.url })
MATCH (s:Starship { url: starship_url })
MERGE (s)-[:APPEARED_IN]->(f);

CALL apoc.load.json($url) YIELD value
UNWIND value.films as film
UNWIND film.vehicles as vehicle_url
MATCH (f:Film { url: film.url })
MATCH (v:Vehicle { url: vehicle_url })
MERGE (v)-[:APPEARED_IN]->(f);



CALL apoc.load.json($url) YIELD value
UNWIND value.people as character
UNWIND character.species as species_url
MATCH (c:Character { url: character.url })
MATCH (spec:Species { url: species_url })
MERGE (c)-[:OF]->(spec);

CALL apoc.load.json($url) YIELD value
UNWIND value.people as character
MATCH (c:Character { url: character.url })
MATCH (p:Planet { url: character.homeworld })
MERGE (c)-[:HOMEWORLD]->(p);

CALL apoc.load.json($url) YIELD value
UNWIND value.people as character
UNWIND character.vehicles as vehicle_url
MATCH (c:Character { url: character.url })
MATCH (v:Vehicle { url: vehicle_url })
MERGE (c)-[:PILOT]->(v);


CALL apoc.load.json($url) YIELD value
UNWIND value.people as character
UNWIND character.starships as starship_url
MATCH (c:Character { url: character.url })
MATCH (s:Starship { url: starship_url })
MERGE (c)-[:PILOT]->(s);


CALL apoc.load.json($url) YIELD value
UNWIND value.species as species
MATCH (spec:Species { url: species.url })
MATCH (p:Planet { url: species.homeworld })
MERGE (spec)-[:HOMEWORLD]->(p);




MATCH (f:Film)
WHERE f.episode_id = 1
SET f.in_universe_year_GT = -32
SET f.in_universe_year = -32;


MATCH (f2:Film)
WHERE f2.episode_id = 2
SET f2.in_universe_year_GT = -22
SET f2.in_universe_year = -22;


MATCH (f3:Film)
WHERE f3.episode_id = 3
SET f3.in_universe_year_GT = -19
SET f3.in_universe_year = -19;


MATCH (f4:Film)
WHERE f4.episode_id = 4
SET f4.in_universe_year_GT = 0
SET f4.in_universe_year = 0;


MATCH (f5:Film)
WHERE f5.episode_id = 5
SET f5.in_universe_year_GT = 3
SET f5.in_universe_year = 3;


MATCH (f6:Film)
WHERE f6.episode_id = 6
SET f6.in_universe_year_GT = 4
SET f6.in_universe_year = 4;


MATCH (f7:Film)
WHERE f7.episode_id = 7
SET f7.in_universe_year_GT = 34
SET f7.in_universe_year = 34;





MATCH (n:Character {name: "Luke Skywalker"})        SET n.death_year_GT = 34   ,n.death_year = 34      , n.death_note = " Dies in The Last Jedi";
MATCH (n:Character {name: "Darth Vader"})           SET n.death_year_GT = 4   ,n.death_year = 4           , n.death_note = " Dies in Return of the Jedi";
MATCH (n:Character {name: "Leia Organa"})           SET n.death_year_GT = 35   ,n.death_year = 35          , n.death_note = " Dies in The Rise of Skywalker";
MATCH (n:Character {name: "Owen Lars"})             SET n.death_year_GT = 0   ,n.death_year = 0           , n.death_note = " Killed by stormtroopers";
MATCH (n:Character {name: "Beru Whitesun Lars"})    SET n.death_year_GT = 0   ,n.death_year = 0           , n.death_note = " Same as above";
MATCH (n:Character {name: "Biggs Darklighter"})     SET n.death_year_GT = 0   ,n.death_year = 0           , n.death_note = " Dies in Battle of Yavin";
MATCH (n:Character {name: "Obi-Wan Kenobi"})        SET n.death_year_GT = 0   ,n.death_year = 0           , n.death_note = " Killed by Darth Vader";
MATCH (n:Character {name: "Anakin Skywalker"})      SET n.death_year_GT = 4   ,n.death_year = 4           , n.death_note = " Same as Vader";
MATCH (n:Character {name: "Wilhuff Tarkin"})        SET n.death_year_GT = 0   ,n.death_year = 0           , n.death_note = " Dies on Death Star";
MATCH (n:Character {name: "Greedo"})                SET n.death_year_GT = 0   ,n.death_year = 0           , n.death_note = " Shot by Han Solo";
MATCH (n:Character {name: "Jabba Desilijic Tiure"}) SET n.death_year_GT = 4   ,n.death_year = 4           , n.death_note = " Killed by Leia";
MATCH (n:Character {name: "Jek Tono Porkins"})      SET n.death_year_GT = 0   ,n.death_year = 0           , n.death_note = " Dies in Battle of Yavin";
MATCH (n:Character {name: "Yoda"})                  SET n.death_year_GT = 4   ,n.death_year = 4           , n.death_note = " Dies of old age";
MATCH (n:Character {name: "Palpatine"})             SET n.death_year_GT = 4   ,n.death_year = 4    , n.death_note = " Dies in ROTJ, returns, dies again in TROS";
MATCH (n:Character {name: "Han Solo"})              SET n.death_year_GT = 34   ,n.death_year = 34          , n.death_note = " Killed by Kylo Ren";
MATCH (n:Character {name: "Qui-Gon Jinn"})          SET n.death_year_GT =  - 32,n.death_year =  - 32          , n.death_note = " Killed by Darth Maul";
MATCH (n:Character {name: "Shmi Skywalker"})        SET n.death_year_GT =  - 22,n.death_year =  - 22          , n.death_note = " Dies in captivity";
MATCH (n:Character {name: "Darth Maul"})            SET n.death_year_GT =  - 2 ,n.death_year =  - 2         , n.death_note = " Killed by Obi-Wan in Rebels";
MATCH (n:Character {name: "Mace Windu"})            SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Presumed dead during Order 66";
MATCH (n:Character {name: "Ki-Adi-Mundi"})          SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Killed during Order 66";
MATCH (n:Character {name: "Kit Fisto"})             SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Killed by Palpatine";
MATCH (n:Character {name: "Eeth Koth"})             SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Killed in Legends, later in canon";
MATCH (n:Character {name: "Adi Gallia"})            SET n.death_year_GT =  - 21,n.death_year =  - 21        , n.death_note = " Killed in Clone Wars by Savage Opress";
MATCH (n:Character {name: "Saesee Tiin"})           SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Killed by Palpatine";
MATCH (n:Character {name: "Yarael Poof"})           SET n.death_year_GT =  - 22,n.death_year =  - 22        , n.death_note = " Dies off-screen";
MATCH (n:Character {name: "Plo Koon"})              SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Order 66 victim";
MATCH (n:Character {name: "Cliegg Lars"})           SET n.death_year_GT =  - 22,n.death_year =  - 22        , n.death_note = " Dies before Episode III";
MATCH (n:Character {name: "Poggle the Lesser"})     SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Executed by Anakin";
MATCH (n:Character {name: "Jango Fett"})            SET n.death_year_GT =  - 22,n.death_year =  - 22          , n.death_note = " Killed by Mace Windu";
MATCH (n:Character {name: "Zam Wesell"})            SET n.death_year_GT =  - 22,n.death_year =  - 22          , n.death_note = " Killed by Jango Fett";
MATCH (n:Character {name: "Dooku"})                 SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Killed by Anakin";
MATCH (n:Character {name: "Padmé Amidala"})         SET n.death_year_GT =  - 19,n.death_year =  - 19          , n.death_note = " Dies giving birth"; 

MATCH ()-[r]->() set r.toDelete=False;
MATCH (n:Character {name: "Darth Vader"})           ,  (f:Film {episode_id:7}) with n,f merge (n)-[:DIED {toDelete:True}]->(f);
MATCH (n:Character {name: "Owen Lars"})             ,  (f:Film {episode_id:4}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Obi-Wan Kenobi"})        ,  (f:Film {episode_id:4}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Anakin Skywalker"})      ,  (f:Film {episode_id:7}) with n,f merge (n)-[:DIED {toDelete:True}]->(f); 
MATCH (n:Character {name: "Wilhuff Tarkin"})        ,  (f:Film {episode_id:4}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Greedo"})                ,  (f:Film {episode_id:4}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Jabba Desilijic Tiure"}) ,  (f:Film {episode_id:6}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Jek Tono Porkins"})      ,  (f:Film {episode_id:4}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Yoda"})                  ,  (f:Film {episode_id:6}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Palpatine"})             ,  (f:Film {episode_id:6}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Han Solo"})              ,  (f:Film {episode_id:6}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Qui-Gon Jinn"})          ,  (f:Film {episode_id:1}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Shmi Skywalker"})        ,  (f:Film {episode_id:2}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Darth Maul"})            ,  (f:Film {episode_id:5}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Mace Windu"})            ,  (f:Film {episode_id:3}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Ki-Adi-Mundi"})          ,  (f:Film {episode_id:3}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Kit Fisto"})             ,  (f:Film {episode_id:3}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Eeth Koth"})             ,  (f:Film {episode_id:3}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Jango Fett"})            ,  (f:Film {episode_id:2}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 
MATCH (n:Character {name: "Padmé Amidala"})         ,  (f:Film {episode_id:3}) with n,f merge (n)-[:DIED {toDelete:false}]->(f); 

MATCH (s:Starship {name:"Sentinel-class landing craft"}) set s.heigth_GT = 2900, s.height = 29 ;
MATCH (s:Starship {name:"Death Star"}) set s.heigth_GT = 12000000, s.height = 120000000;
MATCH (s:Starship {name:"Millennium Falcon"}) set s.heigth_GT = 780, s.height = 78 ;
MATCH (s:Starship {name:"Y-wing"}) set s.heigth_GT = 244, s.height = 244 ;
MATCH (s:Starship {name:"X-wing"}) set s.heigth_GT = 240, s.height = 240 ;
MATCH (s:Starship {name:"TIE Advanced x1"}) set s.heigth_GT = 380, s.height = 380 ;
MATCH (s:Starship {name:"Executor"}) set s.heigth_GT = 126700, s.height = 126700 ;
MATCH (s:Starship {name:"Slave 1"}) set s.heigth_GT = 2150, s.height = 21.50;
MATCH (s:Starship {name:"Imperial shuttle"}) set s.heigth_GT = 2265, s.height = 2265 ;
MATCH (s:Starship {name:"EF76 Nebulon-B escort frigate"}) set s.heigth_GT = 16600, s.height = 16600 ;
MATCH (s:Starship {name:"Calamari Cruiser"}) set s.heigth_GT = 15000, s.height = 15000 ;
MATCH (s:Starship {name:"A-wing"}) set s.heigth_GT = 247, s.height = 247 ;
MATCH (s:Starship {name:"B-wing"}) set s.heigth_GT = 250, s.height = 250;
MATCH (s:Starship {name:"Republic Cruiser"}) set s.heigth_GT = 26800, s.height = 26.800 ;
MATCH (s:Starship {name:"Naboo fighter"}) set s.heigth_GT = 2500, s.height = 25 ;
MATCH (s:Starship {name:"Naboo Royal Starship"}) set s.heigth_GT = 150, s.height = 150 ;
MATCH (s:Starship {name:"Scimitar"}) set s.heigth_GT = 1250, s.height = 1250 ;
MATCH (s:Starship {name:"J-type diplomatic barge"}) set s.heigth_GT = 370, s.height = 370 ;
MATCH (s:Starship {name:"AA-9 Coruscant freighter"}) set s.heigth_GT = 7000, s.height = 7000 ;
MATCH (s:Starship {name:"Jedi starfighter"}) set s.heigth_GT = 244, s.height = 24.4 ;
MATCH (s:Starship {name:"H-type Nubian yacht"}) set s.heigth_GT = 710, s.height = 710 ;
MATCH (s:Starship {name:"Star Destroyer"}) set s.heigth_GT = 26800, s.height = 26800 ;
MATCH (s:Starship {name:"Trade Federation cruiser"}) set s.heigth_GT = 102877, s.height = 102877 ;
MATCH (s:Starship {name:"Theta-class T-2c shuttle"}) set s.heigth_GT = 18500, s.height = 18500 ;
MATCH (s:Starship {name:"T-70 X-wing fighter"}) set s.heigth_GT = 240, s.height = 24 ;
MATCH (s:Starship {name:"Rebel transport"}) set s.heigth_GT = 500, s.height = 500 ;
MATCH (s:Starship {name:"Droid control ship"}) set s.heigth_GT = 102877, s.height = 102877  ;
MATCH (s:Starship {name:"Republic Assault ship"}) set s.heigth_GT = 20000, s.height = 20000 ;
MATCH (s:Starship {name:"Solar Sailer"}) set s.heigth_GT = 480, s.height = 480 ;
MATCH (s:Starship {name:"Republic attack cruiser"}) set s.heigth_GT = 26800, s.height = 26800 ;
MATCH (s:Starship {name:"Naboo star skiff"}) set s.heigth_GT = 300, s.height = 300 ;
MATCH (s:Starship {name:"Jedi Interceptor"}) set s.heigth_GT = 250, s.height = 250 ;
MATCH (s:Starship {name:"arc-170"}) set s.heigth_GT = 381, s.height = 381 ;
MATCH (s:Starship {name:"Banking clan frigte"}) set s.heigth_GT = 24300, s.height = 24300 ;
MATCH (s:Starship {name:"Belbullab-22 starfighter"}) set s.heigth_GT = 300, s.height = 30 ;
MATCH (s:Starship {name:"V-wing"}) set s.heigth_GT = 584, s.height = 584 ;
MATCH (s:Starship {name:"CR90 corvette"}) set s.heigth_GT = 3260, s.height = 3260 ;



MATCH (killer:Character {name: "Darth Vader"}), (victim:Character {name: "Obi-Wan Kenobi"})
MERGE (killer)-[:KILLED {toDelete:false, film: 6}]->(victim);

MATCH (killer:Character {name: "Darth Vader"}), (victim:Character {name: "Palpatine"})
MERGE (killer)-[:KILLED {toDelete:false, film: 6}]->(victim);

MATCH (killer:Character {name: "Kylo Ren"}), (victim:Character {name: "Han Solo"})
MERGE (killer)-[:KILLED {toDelete:false, film: 6}]->(victim);


MATCH (killer:Character {name: "Anakin Skywalker"}), (victim:Character {name: "Count Dooku"})
MERGE (killer)-[:KILLED {toDelete:True, film: 3}]->(victim);

MATCH (killer:Character {name: "Mace Windu"}), (victim:Character {name: "Jango Fett"})
MERGE (killer)-[:KILLED {toDelete:True, film: 4}]->(victim);

MATCH (killer:Character {name: "Palpatine"}), (victim:Character {name: "Mace Windu"})
MERGE (killer)-[:KILLED {toDelete:false, film: 3}]->(victim);

MATCH (killer:Character {name: "Obi-Wan Kenobi"}), (victim:Character {name: "General Grievous"})
MERGE (killer)-[:KILLED {toDelete:false, film: 3}]->(victim);

MATCH (killer:Character {name: "Leia Organa"}), (victim:Character {name: "Jabba Desilijic Tiure"})
MERGE (killer)-[:KILLED {toDelete:false, film: 6}]->(victim);

MATCH (killer:Character {name: "Anakin Skywalker"}), (victim:Character {name: "Younglings"})
MERGE (killer)-[:KILLED {toDelete:false, film: 3}]->(victim);

MATCH (killer:Character {name: "Jango Fett"}), (victim:Character {name: "Zam Wesell"})
MERGE (killer)-[:KILLED {toDelete:True, film: 2}]->(victim);

MATCH (killer:Character {name: "Lando Calrissian"}), (victim:Character {name: "Death Star II Crew"})
MERGE (killer)-[:KILLED {toDelete:false, film: 6}]->(victim);



MERGE (s:Faction { name:'Galactic Empire'});
MERGE (s:Faction { name: 'Galactic Republic'});
MERGE (s:Faction { name: 'Jedi Order'});
MERGE (s:Faction { name: 'Rebel Alliance'});
MERGE (s:Faction { name: 'Resistance'});
MERGE (s:Faction { name: 'Separatists'});
MERGE (s:Faction { name: 'Bounty Hunter'});
MERGE (s:Faction { name: 'Civilian'});
MERGE (s:Faction { name: 'Cloud City Administration'});
MERGE (s:Faction { name: 'Ewoks'});
MERGE (s:Faction { name: 'First Order'});
MERGE (s:Faction { name: 'Gungan Army'});
MERGE (s:Faction { name: 'Gungan Government'});
MERGE (s:Faction { name: 'Hutt Cartel'});
MERGE (s:Faction { name: 'Independent'});
MERGE (s:Faction { name: 'Independent Trader'});
MERGE (s:Faction { name: 'Jedi Order'});
MERGE (s:Faction { name: 'Kaminoan Government'});
MERGE (s:Faction { name: 'Pau an Government'});
MERGE (s:Faction { name: 'Podracer'});
MERGE (s:Faction { name: 'Royal Naboo Security Forces'});
MERGE (s:Faction { name: 'Sith'});
MERGE (s:Faction { name: 'Wookiee Resistance'});

MATCH (c:Character) set c.name_GT = c.name;
MATCH (c:Faction) set c.name_GT = c.name;
MATCH (c:Vehicle) set c.name_GT = c.name;
MATCH (c:Species) set c.average_height_GT = c.average_height;

MATCH (c:Character {name: 'Luke Skywalker'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Luke Skywalker'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'C-3PO'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'C-3PO'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'C-3PO'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'R2-D2'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'R2-D2'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'R2-D2'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Darth Vader'}),(o:Faction {name: 'Sith'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Darth Vader'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Leia Organa'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Leia Organa'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Owen Lars'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Beru Whitesun lars'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'R5-D4'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Biggs Darklighter'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Obi-Wan Kenobi'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Obi-Wan Kenobi'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Anakin Skywalker'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Anakin Skywalker'}),(o:Faction {name: 'Sith'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Wilhuff Tarkin'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Chewbacca'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Chewbacca'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Han Solo'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Han Solo'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Greedo'}),(o:Faction {name: 'Bounty Hunter'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Jabba Desilijic Tiure'}),(o:Faction {name: 'Hutt Cartel'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Wedge Antilles'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Jek Tono Porkins'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Yoda'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Palpatine'}),(o:Faction {name: 'Sith'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Palpatine'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Boba Fett'}),(o:Faction {name: 'Bounty Hunter'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'IG-88'}),(o:Faction {name: 'Bounty Hunter'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Bossk'}),(o:Faction {name: 'Bounty Hunter'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Lando Calrissian'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Lobot'}),(o:Faction {name: 'Cloud City Administration'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Ackbar'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Mon Mothma'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Arvel Crynyd'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Wicket Systri Warrick'}),(o:Faction {name: 'Ewoks'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Nien Nunb'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Qui-Gon Jinn'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Nute Gunray'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Finis Valorum'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Jar Jar Binks'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Roos Tarpals'}),(o:Faction {name: 'Gungan Army'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Rugor Nass'}),(o:Faction {name: 'Gungan Government'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Ric Olié'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Watto'}),(o:Faction {name: 'Independent Trader'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Sebulba'}),(o:Faction {name: 'Podracer'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Quarsh Panaka'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Shmi Skywalker'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Darth Maul'}),(o:Faction {name: 'Sith'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Bib Fortuna'}),(o:Faction {name: 'Hutt Cartel'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Ayla Secura'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Dud Bolt'}),(o:Faction {name: 'Podracer'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Gasgano'}),(o:Faction {name: 'Podracer'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Ben Quadinaros'}),(o:Faction {name: 'Podracer'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Mace Windu'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Ki-Adi-Mundi'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Kit Fisto'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Eeth Koth'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Adi Gallia'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Saesee Tiin'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Yarael Poof'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Plo Koon'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Mas Amedda'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Mas Amedda'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Gregar Typho'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Cordé'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Cliegg Lars'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Poggle the Lesser'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Luminara Unduli'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Barriss Offee'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Dormé'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Dooku'}),(o:Faction {name: 'Sith'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Dooku'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Bail Prestor Organa'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Bail Prestor Organa'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Jango Fett'}),(o:Faction {name: 'Bounty Hunter'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Zam Wesell'}),(o:Faction {name: 'Bounty Hunter'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Dexter Jettster'}),(o:Faction {name: 'Independent'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Lama Su'}),(o:Faction {name: 'Kaminoan Government'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Taun We'}),(o:Faction {name: 'Kaminoan Government'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Jocasta Nu'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Ratts Tyerell'}),(o:Faction {name: 'Podracer'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'R4-P17'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Wat Tambor'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'San Hill'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Shaak Ti'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Grievous'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Tarfful'}),(o:Faction {name: 'Wookiee Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Raymus Antilles'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Sly Moore'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Sly Moore'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Tion Medon'}),(o:Faction {name: 'Pau an Government'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Finn'}),(o:Faction {name: 'First Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Finn'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Rey'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Rey'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Poe Dameron'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'BB8'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Captain Phasma'}),(o:Faction {name: 'First Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Character {name: 'Padmé Amidala'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);

MATCH (c:Vehicle {name: 'Sand Crawler'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'T-16 skyhopper'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'X-34 landspeeder'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'TIE/LN starfighter'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Snowspeeder'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'TIE bomber'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'AT-AT'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'AT-ST'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Storm IV Twin-Pod cloud car'}),(o:Faction {name: 'Cloud City Administration'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Sail barge'}),(o:Faction {name: 'Hutt Cartel'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Bantha-II cargo skiff'}),(o:Faction {name: 'Hutt Cartel'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'TIE/IN interceptor'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Imperial Speeder Bike'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Vulture Droid'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Multi-Troop Transport'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Armored Assault Tank'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Single Trooper Aerial Platform'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'C-9979 landing craft'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Tribubble bongo'}),(o:Faction {name: 'Gungan Army'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Sith speeder'}),(o:Faction {name: 'Sith'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Zephyr-G swoop bike'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Koro-2 Exodrive airspeeder'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Koro-2 Exodrive airspeeder'}),(o:Faction {name: 'Naboo Elite'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'XJ-6 airspeeder'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'XJ-6 airspeeder'}),(o:Faction {name: 'Coruscant'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'LAAT/i'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'LAAT/c'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Tsmeu-6 Characteral wheel bike'}),(o:Faction {name: 'General Grievous'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Tsmeu-6 Characteral wheel bike'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Emergency Firespeeder'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Droid tri-fighter'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Oevvaor jet catamaran'}),(o:Faction {name: 'Wookiee Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Raddaugh Gnasp fluttercraft'}),(o:Faction {name: 'Wookiee Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Clone turbo tank'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Corporate Alliance tank droid'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Droid gunship'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'AT-RT'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'AT-TE'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'SPHA'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Flitknot speeder'}),(o:Faction {name: 'Wookiee Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Neimoidian shuttle'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Vehicle {name: 'Geonosian starfighter'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);


MATCH (c:Starship {name: 'Sentinel-class landing craft'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Death Star'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Millennium Falcon'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Y-wing'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'X-wing'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'TIE Advanced x1'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Executor'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Slave 1'}),(o:Faction {name: 'Bounty Hunter'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Imperial shuttle'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'EF76 Nebulon-B escort frigate'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Calamari Cruiser'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'A-wing'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'B-wing'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Republic Cruiser'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Naboo fighter'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Naboo Royal Starship'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Scimitar'}),(o:Faction {name: 'Sith'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'J-type diplomatic barge'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'AA-9 Coruscant freighter'}),(o:Faction {name: 'Civilian'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Jedi starfighter'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'H-type Nubian yacht'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Star Destroyer'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Trade Federation cruiser'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Theta-class T-2c shuttle'}),(o:Faction {name: 'Galactic Empire'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'T-70 X-wing fighter'}),(o:Faction {name: 'Resistance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Rebel transport'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Droid control ship'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Republic Assault ship'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Solar Sailer'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Republic attack cruiser'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Naboo star skiff'}),(o:Faction {name: 'Royal Naboo Security Forces'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Jedi Interceptor'}),(o:Faction {name: 'Jedi Order'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'arc-170'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Banking clan frigte'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'Belbullab-22 starfighter'}),(o:Faction {name: 'Separatists'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'V-wing'}),(o:Faction {name: 'Galactic Republic'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);
MATCH (c:Starship {name: 'CR90 corvette'}),(o:Faction {name: 'Rebel Alliance'}) with c,o MERGE (c) -[r:BELONGS_TO {toDelete:False}]-> (o);





MATCH (sp:Species)<-[r1:OF]-(c:Character)-[r2:PILOT]->(s:Starship) where apoc.number.parseInt(sp.average_height)>s.height 
WITH sp, c, s, r1, r2, apoc.create.uuid() AS uuid
SET c.isViolation = true,
      sp.isViolation = true,
      s.isViolation = true,
      r1.isViolation = true,
      r2.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      s.violationId = coalesce(s.violationId, []) + uuid,
      sp.violationId = coalesce(sp.violationId, []) + uuid,
      r1.violationId = coalesce(r1.violationId, []) + uuid,
      r2.violationId = coalesce(r2.violationId, []) + uuid;
 


MATCH (f1:Film)<-[r1:APPEARED_IN]-(c:Character)-[r2:DIED]->(f:Film) where f1.in_universe_year>f.in_universe_year
WITH f1, c, f, r1, r2, apoc.create.uuid() AS uuid
SET c.isViolation = true,
      f1.isViolation = true,
      f.isViolation = true,
      r1.isViolation = true,
      r2.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      f.violationId = coalesce(f.violationId, []) + uuid,
      f1.violationId = coalesce(f1.violationId, []) + uuid,
      r1.violationId = coalesce(r1.violationId, []) + uuid,
      r2.violationId = coalesce(r2.violationId, []) + uuid;

MATCH (c:Character)-[r1:DIED]->(f:Film) where c.death_year<>f.in_universe_year
WITH  c, f, r1, apoc.create.uuid() AS uuid
SET c.isViolation = true,
      f.isViolation = true,
      r1.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      f.violationId = coalesce(f.violationId, []) + uuid,
      r1.violationId = coalesce(r1.violationId, []) + uuid;

MATCH (c:Character)-[r1:KILLED]->(c1:Character)-[r2:DIED]->(f:Film), (f1:Film) where f1.episode_id=r1.film and f.in_universe_year<>c1.death_year 
WITH  c, c1, f,f1, r1, r2, apoc.create.uuid() AS uuid
SET c.isViolation = true,
      c1.isViolation = true,
      f.isViolation = true,
      f1.isViolation = true,
      r1.isViolation = true,
      r2.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      c1.violationId = coalesce(c1.violationId, []) + uuid,
      f.violationId = coalesce(f.violationId, []) + uuid,
      f1.violationId = coalesce(f1.violationId, []) + uuid,
      r1.violationId = coalesce(r1.violationId, []) + uuid,
      r2.violationId = coalesce(r2.violationId, []) + uuid;
