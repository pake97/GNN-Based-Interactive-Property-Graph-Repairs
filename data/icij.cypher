match ()-[r]->() set r.toDelete=False;
MATCH (o:Officer)
SET o.networth = toInteger(rand() * 90000000) + 10000000
RETURN o.name, o.networth;

MATCH (o:Officer)
SET o.networth_GT = o.networth;

//create 200K violations
CALL apoc.periodic.iterate(
  "
  MATCH (a:Address)-[p:REGISTERED_ADDRESS]-(b)
  WHERE apoc.text.indexOf(b.country_codes, a.country_codes) < 0
  RETURN DISTINCT a, b, p
  ",
  "
  WITH a, b, p, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET a.isViolation = true,
      b.isViolation = true,
      p.isViolation = true,

      a.violationId = coalesce(a.violationId, []) + uuid,
      b.violationId = coalesce(b.violationId, []) + uuid,
      p.violationId = coalesce(p.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    set p.toDelete=True
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET b.country_codes = b.country_codes + ',' + a.country_codes
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


//138k violations
CALL apoc.periodic.iterate(
  "
  MATCH (o)-[of:officer_of]->(e:Entity)
        -[ra1:registered_address]->(a:Address)<-[ra2:registered_address]-(o:Officer)
  RETURN DISTINCT o, e, a, of, ra1, ra2
  LIMIT 15000
  ",
  "
  WITH o, e, a, of, ra1, ra2, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET o.isViolation = true,
      e.isViolation = true,
      a.isViolation = true,
      of.isViolation = true,
      ra1.isViolation = true,
      ra2.isViolation = true,

      o.violationId = coalesce(o.violationId, []) + uuid,
      e.violationId = coalesce(e.violationId, []) + uuid,
      a.violationId = coalesce(a.violationId, []) + uuid,
      of.violationId = coalesce(of.violationId, []) + uuid,
      ra1.violationId = coalesce(ra1.violationId, []) + uuid,
      ra2.violationId = coalesce(ra2.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET ra1.toDelete_GT = true
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET ra2.toDelete_GT = true
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


//324
CALL apoc.periodic.iterate(
  "
  MATCH (e:Entity)<-[inter:intermediary_of]-(i:Intermediary)
        -[off:officer_of {link: 'shareholder of'}]->(e)
  RETURN DISTINCT e, i, inter, off
  ",
  "
  WITH e, i, inter, off, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET e.isViolation = true,
      i.isViolation = true,
      inter.isViolation = true,
      off.isViolation = true,

      e.violationId = coalesce(e.violationId, []) + uuid,
      i.violationId = coalesce(i.violationId, []) + uuid,
      inter.violationId = coalesce(inter.violationId, []) + uuid,
      off.violationId = coalesce(off.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET inter.toDelete_GT = true
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET off.toDelete_GT = true
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);




CALL apoc.periodic.iterate(
  "
  MATCH (o:Officer)-[of:officer_of {link: 'shareholder of'}]->(e:Entity)
  WHERE o.networth < 17000000
  RETURN DISTINCT o, e, of
  ",
  "
  WITH o, e, of, apoc.create.uuid() AS uuid, rand() AS randomValue, 17000000 + toInteger(rand() * 1000000) AS newWorth

  SET o.isViolation = true,
      e.isViolation = true,
      of.isViolation = true,

      o.violationId = coalesce(o.violationId, []) + uuid,
      e.violationId = coalesce(e.violationId, []) + uuid,
      of.violationId = coalesce(of.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET o.networth_GT = newWorth
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET of.toDelete_GT = true
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);





