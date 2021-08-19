select * from product;
-- Part 1
-- 2.1
select product_name, regist_date
from product
where regist_date > '2009-04-28';
-- 2.2
-- (1)
/*SELECT *
  FROM product
 WHERE purchase_price = NULL;
 
 从product里选取符合没有售价的row的全部字段*/
 -- (2)
/*SELECT *
  FROM product
 WHERE purchase_price <> NULL;
 
 因为跟NULL对比时 结果总是NUll 所以跟第一是一样的*/
-- (3)
/* SELECT *
  FROM product
 WHERE product_name > NULL;
 
 因为跟NULL对比时 结果总是NUll 所以跟第一是一样的*/
 

 -- 2.3
 select product_name, sale_price, purchase_price from product
 where sale_price - purchase_price >=500;
 
 select product_name, sale_price, purchase_price from product
 where NOT sale_price < purchase_price +500;
 
 
 -- 2.4
 select product_name, product_type, sale_price * 0.9-purchase_price as profit from product
 where sale_price * 0.9-purchase_price > 100;
 
-- Part 2
-- 2.5
/* SELECT product_id, SUM（product_name）
-- 本SELECT语句中存在错误。FROM product 
 GROUP BY product_type 
 WHERE regist_date > '2009-09-01' 
 
 1. product_name是字符 因此没有办法SUM
 2. 没有选择目标数据库 from product
 2. Group by应该在Where后面*/

-- 2.6
select product_type, sum(sale_price) as sum, sum(purchase_price) as sum from product
group by product_type having sum(sale_price) > sum(purchase_price) *1.5;

-- 2.7
-- MySQL 认为NULL比任何non-NULL都小 如果在其他的平台就ok了？
select * from product
order by regist_date DESC, sale_price;