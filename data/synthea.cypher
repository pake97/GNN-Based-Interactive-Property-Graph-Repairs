match ()-[r]->() set r.toDelete = False;

CALL apoc.periodic.iterate(
  "
  MATCH (p:Patient)-[startRel:INSURANCE_START]->(payer1:Payer),
        (p)-[endRel:INSURANCE_END]->(payer2:Payer)
  WHERE payer1.id <> payer2.id
  RETURN DISTINCT p, payer1, payer2, startRel, endRel
  ",
  "
  WITH p, payer1, payer2, startRel, endRel, apoc.create.uuid() AS uuid

  SET p.isViolation = true,
      payer1.isViolation = true,
      payer2.isViolation = true,
      startRel.isViolation = true,
      endRel.isViolation = true,

      p.violationId = coalesce(p.violationId, []) + uuid,
      payer1.violationId = coalesce(payer1.violationId, []) + uuid,
      payer2.violationId = coalesce(payer2.violationId, []) + uuid,
      startRel.violationId = coalesce(startRel.violationId, []) + uuid,
      endRel.violationId = coalesce(endRel.violationId, []) + uuid,

      endRel.toDelete = true
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


MATCH (e:Encounter)-[r:HAS_END]->(e1:Encounter)
WHERE e1.isEnd = true
 with e1 
 limit 400 set e1.isEnd_GT = true, e1.isEnd = False;


 CALL apoc.periodic.iterate(
  "
  MATCH (e:Encounter)-[r:HAS_END]->(e2:Encounter)
  WHERE e2.isEnd = false
  RETURN DISTINCT e, e2, r
  ",
  "
  WITH e, e2, r, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET e.isViolation = true,
      e2.isViolation = true,
      r.isViolation = true,

      e.violationId = coalesce(e.violationId, []) + uuid,
      e2.violationId = coalesce(e2.violationId, []) + uuid,
      r.violationId = coalesce(r.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET e2.isEnd_GT = false
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    set r.toDelete=True
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
)



MATCH (p:Payer)
WITH p LIMIT 200
WITH p, apoc.create.uuid() AS uuid
CREATE (p)-[r:HAS_PAYER]->(p)
SET p.isViolation = true,
    p.violationId = coalesce(p.violationId, []) + uuid,
    r.isViolation = true,
    r.toDelete = true,
    r.violationId = coalesce(r.violationId, []) + uuid;
